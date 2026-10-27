# 源码核验与边界

OBSERVED：当前投机PromptCache仅缓存model.encode_prompt的context与mask（投机采样/fastwam-base/cached-speedcode-v1/prompt_cache.py:7–46）。它没有缓存action text_embedding或30层cross KV。

官方固定源：wam_tutorial_a100/upstream/fastwam/src/fastwam/models/wan22/。
- fastwam.py:224–245：文本context后追加当前proprio token，故跨chunk不宜按相同prompt直接复用全部KV。
- fastwam.py:703–736：每次action denoise仍调用action_expert.prepare和完整30层action计算。
- action_dit.py:312–327：prepare中context_emb=text_embedding(context)，不依赖当前action latent或timestep。
- wan_video_dit.py:199–221：cross attention的K=norm_k(k(ctx))，V=v(ctx)；Q来自当前action状态。

本轮实现：optimized/cross_cache.py创建一次policy请求范围内的ChunkCrossCache；每个实际batch独立计算condition和30层K/V。CachedTorchBlock及CachedAction保留官方数学。optimized/csrc/fastwam_persistent.cuh只在显式cross_cached=True时跳过stage11/13及其阶段屏障；缓存入口已有normalized K和V。五类GEMM的TMA/1T4M3E/mbarrier不改，FFN2连续FP32仍关闭。原persistent五参数接口兼容，默认cross_cached=False。

六组真实层验证已发现B1/B7条件投影结果有数值差异，故禁止把B1 KV广播给B7；在本轮实际policy查询中B1/B7分别建立后，接受与动作逐位核验通过。仅此有限样本，不能推广成所有输入的数学逐位定理。

共同缓存为工程基线；它不是候选调度创新。新缓存compile改变图划分，须单列与原compile数值差异。四项科学权限不变。

本轮不改变全局策略、不重训、不校准、不修改其他会话运行文件/队列。不构成机器人环境成功率或自然任务E2E实测。
