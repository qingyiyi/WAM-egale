// Derived byte-for-byte helper bodies from pinned PyTorch CUDA LayerNorm.
// commit 0fabc3ba44823f257e70ce397d989c8de5e362c1; BSD license beside file.
#pragma once
#include <ATen/AccumulateType.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAMathCompat.h>
namespace wam_block_norm {
using at::acc_type;
constexpr int vec_size = 4;
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

struct WelfordDataLN{
  float mean;
  float sigma2;
  float count;
  C10_HOST_DEVICE WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  C10_HOST_DEVICE WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U, bool rms_norm> __device__
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  if constexpr (!rms_norm){
    U delta = val - curr_sum.mean;
    U new_count = curr_sum.count + 1.f;
    U new_mean = curr_sum.mean + delta * (1.f/new_count); //proper division is slow, this is less accurate but noticeably faster
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
  } else{
    return {0.f, curr_sum.sigma2 + val * val, 0};
  }
}

template<bool rms_norm> __device__
WelfordDataLN cuWelfordCombine(
  const WelfordDataLN dataB,
  const WelfordDataLN dataA
) {
  if constexpr (!rms_norm){
    using U = decltype(dataB.count);
    U delta = dataB.mean - dataA.mean;
    U count = dataA.count + dataB.count;
    U mean, sigma2;
    if (count > decltype(dataB.count){0}) {
      auto coef = 1.f/count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
      auto nA = dataA.count * coef;
      auto nB = dataB.count * coef;
      mean = nA*dataA.mean + nB*dataB.mean;
      sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
      mean = U(0);
      sigma2 = U(0);
    }
    return {mean, sigma2, count};
  } else {
    return {0.f, dataB.sigma2 + dataA.sigma2, 0};
  }
}

template<typename T, bool rms_norm = false>
__device__ WelfordDataLN compute_stats(
  const T*  __restrict__ X,
  const int N,
  float * buf
  ) {
    //X points to the row to read
    using vec_t = aligned_vector<T, vec_size>;
    using acc_t = acc_type<T, true>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(X);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;
    WelfordDataLN wd(0.f, 0.f, 0.f);
    //no tail, we check that N is multiple of vec_size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++){
        wd = cuWelfordOnlineSum<acc_t, rms_norm>(static_cast<acc_t>(data.val[ii]), wd);
      }
    }
    // intra-warp reduction
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset), WARP_SHFL_DOWN(wd.sigma2, offset), WARP_SHFL_DOWN(wd.count, offset)};
        wd = cuWelfordCombine<rms_norm>(wd, wdB);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float * meansigmabuf = buf;
      float * countbuf = buf + blockDim.y;
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          meansigmabuf[2*wrt_y] = wd.mean;
          meansigmabuf[2*wrt_y+1] = wd.sigma2;
          countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          WelfordDataLN wdB{meansigmabuf[2*threadIdx.y],
                          meansigmabuf[2*threadIdx.y+1],
                          countbuf[threadIdx.y]};
          wd = cuWelfordCombine<rms_norm>(wd, wdB);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0 && threadIdx.y ==0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2/float(N);
      }
      __syncthreads();
      return WelfordDataLN{meansigmabuf[0], meansigmabuf[1],0.f};

    } else {
      return WelfordDataLN{WARP_SHFL(wd.mean,0), WARP_SHFL(wd.sigma2,0)/float(N), 0.f};
    }
}


template <typename T, typename T_ACC, bool rms_norm = false,
typename std::enable_if_t<!std::is_same_v<T, double>, int> = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
    extern __shared__ float s_data[]; //if we made smem WelfordDataLN type, there would be bank conflicts,
    //as one thread would have to write 3 consecutive floats
    auto i1 = blockIdx.x;
    const T * block_row = X + i1 * N;
    WelfordDataLN wd = compute_stats<T, rms_norm>(block_row, N, s_data);

    using vec_t = aligned_vector<T, vec_size>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(block_row);
    const vec_t * gamma_vec = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    const vec_t * beta_vec = (beta != nullptr) ? reinterpret_cast<const vec_t*>(beta) : nullptr;
    vec_t * Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;

    T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);

    // No tail, N is guaranteed to be multiple of vec size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;

      // Computation is performed in T_ACC, X is cast to T_ACC and result is implicitly cast to T
      if (gamma_vec != nullptr && beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          if constexpr (!rms_norm){
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
              + static_cast<T_ACC>(beta_vec[i].val[ii]);
          } else {
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * static_cast<T_ACC>(data.val[ii]));
          }
        }
      } else if (gamma_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          if constexpr (!rms_norm){
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
          } else {
            out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * static_cast<T_ACC>(data.val[ii]));
          }
        }
      } else if (beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
            out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          if constexpr (!rms_norm){
            out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
          } else {
            out.val[ii] = rstd_val * static_cast<T_ACC>(data.val[ii]);
          }
        }
      }
      Y_vec[i] = out;
    }
    if (thrx == 0) {
      if constexpr (!rms_norm){
        mean[i1] = wd.mean;
      }
      rstd[i1] = rstd_val;
    }
}

template <typename T, typename T_ACC, bool rms_norm = false,
typename std::enable_if_t<std::is_same_v<T, double>, int> = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int /*N*/,
  T_ACC /*eps*/,
  const  T* __restrict__ /*X*/,
  const  T* /*gamma*/,
  const  T* /*beta*/,
  T_ACC* /*mean*/,
  T_ACC* /*rstd*/,
  T* /*Y*/){
    CUDA_KERNEL_ASSERT(false && "doesn't work with double");
  }

//to avoid windows SFINAE errors
template <typename T, typename T_ACC, bool rms_norm = false>
__global__ void vectorized_layer_norm_kernel(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
    vectorized_layer_norm_kernel_impl<T, T_ACC, rms_norm>(N, eps, X, gamma, beta, mean, rstd, Y);
  }


} // namespace wam_block_norm
