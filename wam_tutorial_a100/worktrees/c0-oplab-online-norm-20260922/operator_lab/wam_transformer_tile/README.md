# 唯一 CUDA SDPA baseline

仅复现当前 A100/SM80、BF16、contiguous `[B,S,H,128]`、inference、noncausal/no-mask/no-dropout 合同；不支持完整PyTorch SDPA API，也没有内置SDPA fallback。

- 算子：`csrc/standalone_flash_sdpa.cu`，公共入口 `wam_transformer_tile.cuda_sdpa_standalone(q,k,v,scale=None)`，直接native导出；默认scale在C++解析，原逐项输入检查保留。
- 唯一测试入口：`test_standalone_sdpa.py`。`contract.py`/`fixtures.py`只用于原full-block与真实fixture replay，Torch SDPA只作为oracle。
- 源码对齐：实际PyTorch2.9.0+cu126 commit `0fabc3ba44823f257e70ce397d989c8de5e362c1` 的固定FlashAttention/CUTLASS；必要vendor头文件、LICENSE和来源JSON保留。nvcc12.4编译，**不声称与Torch wheel具有相同SASS**。
- 当前版本固定两独立进程，每进程全部六真实短/长K2160输入、7轮：原`.02/.02`正确性及所有eager/core/full-block中位CUDA/PyTorch<=1通过。只对这份固定范围负责，不是所有形状/机器、每个单次调用或WAM/LIBERO E2E承诺。
- 上一版full-block慢0.4013%、其他历史性能/计时失败与原raw全部归档保留；主机接口变更不是历史失败唯一因果证明。没有修改投机采样、共享模型/server或系统时钟。

## 构建（现有环境，不安装）

```bash
cd "/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile"
CUDA_HOME=/usr/local/cuda-12.4 CC=/rjs/cxz/miniconda3/bin/x86_64-conda-linux-gnu-cc CXX=/rjs/cxz/miniconda3/bin/x86_64-conda-linux-gnu-c++ TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=1 PYTHONDONTWRITEBYTECODE=1 /rjs/cxz/miniconda3/envs/va/bin/python setup.py build_ext --inplace
```

## 原测试

GPU测试需确认闲置，默认参数和原门槛不要覆盖：

```bash
cd "/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile"
PYTHONDONTWRITEBYTECODE=1 /rjs/cxz/miniconda3/envs/va/bin/python test_standalone_sdpa.py "/rjs/cxz/论文探索/wam_tutorial_a100/oplab_outputs/c0-real-fixture-libero-k0-20260925/fixtures/block0_pos_b2_s128.pt" "/rjs/cxz/论文探索/wam_tutorial_a100/oplab_outputs/c0-real-fixture-libero-k0-20260925/fixtures/block0_pos_b2_s16.pt" "/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-real-k2160-par2-20261011-122816/workers/tile0/fixtures/block0_pos_b2_s128.pt" "/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-real-k2160-par2-20261011-122816/workers/tile0/fixtures/block0_pos_b2_s16.pt" "/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-real-k2160-par2-20261011-122816/workers/tile1/fixtures/block0_pos_b2_s128.pt" "/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-real-k2160-par2-20261011-122816/workers/tile1/fixtures/block0_pos_b2_s16.pt" --require-parity --output-json /absolute/new/report.json
```

CPU只查source/build/fixture metadata，不是GPU正确性或性能：在上面的命令加`--cpu-source-checks`，并设置`CUDA_VISIBLE_DEVICES=''`。

本次已用完固定两次GPU预算，不会因单轮/历史负数再测同版本挑通过。原六fixture不包含在代码包中、不合成补KV。

验收与完整raw：[独立tile HW](../../../../../.pipeline/research/tile-sdpa/PYTORCH29_FINAL_20261012.md)、[r1](../../../../../review/OMISSIONS.md)、[r2](../../../../../review/OMISSIONS.md)、[完整离线审计](../../outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/actual-release-report-audit.json)。退役tile采集/旧头/旧构建字节：[精确归档](../../../../../review/OMISSIONS.md)；历史输出源码里仍记录当时的原路径，不作为当前live依赖。
