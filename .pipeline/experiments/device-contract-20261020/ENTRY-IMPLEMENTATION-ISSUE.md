# 独立入口依赖路径实施失败

首次metadata-1在调用第一个完整block之前遇到ImportError：冻结baseline DSO链接本机绝对路径的attention DSO。0完整block/0heuristic；并非数值失败。实际CUDA context存在约该worker生命周期10.83秒（CUDA active时间UNKNOWN），root真实reap、NVML为空。保留原STATUS、RESULT及receipt。

修正只在本轮独立脚本将debug参考入口换为已在远端部署的optimized NativeBlock(mode=sequential)，数学/DSO本身不修改；该顺序链既有原Torch有限逐位证据。直接Q/FFN2金标准仍是同输入原Torch Linear。本机baseline、原模型与依赖不变。

不是完整机器人任务重试；没有重复已经完成的数值候选，也没有删除失败。新entry-corrected-1独立命名，沿用最初T0、10:01:56北京工作截止/10:11:56清理和总8block/120 primitive上限，原耗时合并记账。只修复具体路径入口，不扩大配置或研究臂。
