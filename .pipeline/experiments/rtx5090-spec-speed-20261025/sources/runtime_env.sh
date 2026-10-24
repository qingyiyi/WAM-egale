#!/usr/bin/env bash
set -eu
source /root/autodl-tmp/cxz/activate-wam-sm120.sh
export WAM_SPEED_ROOT=/root/autodl-tmp/cxz/spec-speed-20261025
export TMPDIR="$WAM_SPEED_ROOT/tmp" TMP="$WAM_SPEED_ROOT/tmp" TEMP="$WAM_SPEED_ROOT/tmp"
export HF_HOME="$WAM_SPEED_ROOT/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_CACHE_PATH="$WAM_SPEED_ROOT/cache/cuda" TORCH_EXTENSIONS_DIR="$WAM_SPEED_ROOT/cache/torch_extensions" TRITON_CACHE_DIR="$WAM_SPEED_ROOT/cache/triton"
export XDG_CACHE_HOME="$WAM_SPEED_ROOT/cache/xdg" PYTHONPYCACHEPREFIX="$WAM_SPEED_ROOT/cache/pycache"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MAX_JOBS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH="$WAM_SPEED_ROOT/config/libero"
export PYTHONPATH="$WAM_SPEED_ROOT/port/libero:$WAM_SPEED_ROOT/port/robotwin:$WAM_SPEED_ROOT/sim/LIBERO:$WAM_SPEED_ROOT/bundle/project/投机采样/experiments:$WAM_SPEED_ROOT/bundle/project/投机采样/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PREFIX_ACCEPT_ENABLED=1 PREFIX_ACCEPT_GPU_APPROVED=1
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export VK_ICD_FILENAMES="$WAM_SPEED_ROOT/config/nvidia_icd.json" WAM_RENDER_PCI=0000:38:00.0
export PYTHONPATH="$WAM_SPEED_ROOT/bundle/robotwin/eval:$WAM_SPEED_ROOT/sim/curobo/src:$PYTHONPATH"
export LD_LIBRARY_PATH="$WAM_SPEED_ROOT/deps/glvnd22:$WAM_SPEED_ROOT/deps/vulkan22:$WAM_SPEED_ROOT/client310/lib/python3.10/site-packages/sapien/oidn_library:$LD_LIBRARY_PATH"
export __EGL_VENDOR_LIBRARY_FILENAMES="$WAM_SPEED_ROOT/config/egl_nvidia.json"
export WARP_CACHE_PATH="$WAM_SPEED_ROOT/cache/warp" MPLCONFIGDIR="$WAM_SPEED_ROOT/cache/matplotlib"
export CPLUS_INCLUDE_PATH="$WAM_SPEED_ROOT/client310/include/python3.10" C_INCLUDE_PATH="$WAM_SPEED_ROOT/client310/include/python3.10"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.8
export __GL_SHADER_DISK_CACHE_PATH="$WAM_SPEED_ROOT/cache/graphics" OPTIX_CACHE_PATH="$WAM_SPEED_ROOT/cache/optix" WANDB_MODE=offline
