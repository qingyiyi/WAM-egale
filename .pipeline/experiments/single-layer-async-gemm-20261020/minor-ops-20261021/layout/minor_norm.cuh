// Derived pinned Torch; license in norm.cuh.
#pragma once
namespace wam_optimized_norm {
template <typename T, typename T_ACC, bool rms_norm = false,
typename std::enable_if_t<!std::is_same_v<T, double>, int> = 0>
__device__ __inline__ void minor_rms_forward(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y, float* s_data){
    // supplied scratch //if we made smem WelfordDataLN type, there would be bank conflicts,
    //as one thread would have to write 3 consecutive floats
    auto i1 = 0;
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
      if(rstd)rstd[i1] = rstd_val;
    }
}


}
