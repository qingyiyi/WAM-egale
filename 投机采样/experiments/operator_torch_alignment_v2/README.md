# 独立Torch2.9数值对齐版本v2

来源：原wam_transformer_tile的固定PyTorch/FlashAttention/CUTLASS实现，完整来源与许可证在csrc/vendor。旧目录不变。

本轮唯一数学编译变化：移除--use_fast_math；命名空间/扩展名隔离。未调用Torch SDPA作为自有实现，不改变split/tile/阈值。是否解决整模型误差以实测结果为准。

编译：CUDA_VISIBLE_DEVICES='' CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=1 /rjs/cxz/miniconda3/envs/va/bin/python setup.py build_ext --inplace

CPU编译CUDA不可见。固定Torch2.9.0+cu126；本地nvcc12.4与wheel12.6仍不同，不宣称二进制相同。这里只支持SM80 BF16 D128 inference契约。


2026-10-18验证：9真实保存输入、108 attention、54完整block、6完整30层verifier query、3组短中长完整canonical policy配对输出与声明有效状态均逐位通过。详情见`.pipeline/experiments/operator-torch-alignment-20261018/RESULT.md`。有限输入不证明任意输入或闭环质量；新二进制性能尚未复测。

用法：已有真实server实例可`from integration import installed`，在`with installed(server.transformer):`内执行原infer；完整模型验证实际覆盖该install/restore。独立`server_entry.py`可包装canonical服务，沿用原CLI；该RPC入口本轮未另启动，生产默认保持原值。C++本身检查SM80/BF16/D128/contiguous/inference契约，不支持的输入直接报错，无Torch SDPA fallback。
