# 当前实现阅读顺序

正式代码位于`投机采样/fastwam-base/`，与原LingBot相关源码同级，HW仅放研究记录。各目录为独立冻结部署版本，不能混用数值/速度/质量身份。

|目的|入口|
|---|---|
|完整闭环target与预处理|[fullcode-v1/runtime.py](../../投机采样/fastwam-base/fullcode-v1/runtime.py)，固定官方factory/normalizer，strict加载与低内存T5；[server.py](../../投机采样/fastwam-base/fullcode-v1/server.py)、[client.py](../../投机采样/fastwam-base/fullcode-v1/client.py)|
|官方infer_action等价注入|[adapter.py](../../投机采样/fastwam-base/fullcode-v1/adapter.py)；实际step/delta取scheduler，不从timestep差倒推|
|teacher方向采集|[real_speculative.py](../../投机采样/fastwam-base/fullcode-v1/real_speculative.py)的CanonicalCollector；canonical状态、previous方向、target方向、timestep、delta、has_previous|
|54,094参数方向预测器|[draft.py](../../投机采样/fastwam-base/fullcode-v1/draft.py)、[train.py](../../投机采样/fastwam-base/fullcode-v1/train.py)|
|B1/B7 target adapter、隔离和数值审计|[simulation_reference.py](../../投机采样/fastwam-base/fullcode-v1/simulation_reference.py)；RealSpeculative复用adapter，但实际接受不强制历史深度|
|K2/B2树、FP32打分与真实提交|[real_speculative.py](../../投机采样/fastwam-base/fullcode-v1/real_speculative.py)；score→depth_then_cost→fallback/bonus；当前尾部不足3步canonical|
|PF2、两客户端与全量|[policy_schedule.py](../../投机采样/fastwam-base/fullcode-v1/policy_schedule.py)、[phase_worker.py](../../投机采样/fastwam-base/fullcode-v1/phase_worker.py)、[full_worker.py](../../投机采样/fastwam-base/fullcode-v1/full_worker.py)|
|原始动作/提交/全量审计|[summarize_full.py](../../投机采样/fastwam-base/fullcode-v1/summarize_full.py)、[summarize_triarm.py](../../投机采样/fastwam-base/fullcode-v1/summarize_triarm.py)|
|共同prompt缓存与新速度|[prompt_cache.py](../../投机采样/fastwam-base/cached-speedcode-v1/prompt_cache.py)、[speed_triarm_v2.py](../../投机采样/fastwam-base/cached-speedcode-v1/speed_triarm_v2.py)；与原suite-speedcode-v1分离|
|Robo完成4局canonical的版本|[robotwin-code-v4/runtime.py](../../投机采样/fastwam-base/robotwin-code-v4/runtime.py)、[server.py](../../投机采样/fastwam-base/robotwin-code-v4/server.py)、[client.py](../../投机采样/fastwam-base/robotwin-code-v4/client.py)|
|Robo仅准备的投机/训练|[robotwin-spec-v1/README.md](../../投机采样/fastwam-base/robotwin-spec-v1/README.md)、[verifier.py](../../投机采样/fastwam-base/robotwin-spec-v1/verifier.py)、[train.py](../../投机采样/fastwam-base/robotwin-spec-v1/train.py)；尚未GPU训练/三臂评估|

上游固定版本入口：

- [FastWAM infer_action](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py)
- [action_dit.py](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/action_dit.py)、[mot.py](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py)
- [scheduler_continuous.py](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py)
- [官方LIBERO评估](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/eval_libero_single.py)
- [固定HF资产](https://huggingface.co/yuanty/fastwam/tree/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705)

原部署脚本通过`__file__`相对ROOT定位：需要在隔离run目录放`code/`、固定`source/FastWAM`、匹配的REGISTRATION/PANEL/资产路径和results，不能直接在公开源码目录随意运行。GPU runner遵循既有截止/资源登记，本公开包未导出私有SSH调度器、资产或完整环境。现有原注册已冻结且有历史截止，不是新运行授权；CPU统计核验可直接运行本目录verify脚本。

不要把源码目录叫`simulation_reference`理解为本轮强制80%模拟：真实RealSpeculative的六个分数来自target，决定实际深度；旧强制Long轨迹模拟结果另存历史，不是新速度数字。算子最新证据见[Cross-KV报告](../../.pipeline/experiments/fastwam-crosskv-cache-20261027/REVIEW_RETURN.md)，本次未导出正在演进的完整算子开发树或二进制，也未将其并入投机默认。
