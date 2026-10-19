# 算子代码入口已迁移

当前只维护[算子研究](../../算子研究/README.md)的baseline/optimized两份实现，共享安装、输入、数值验证与计时工具。主源码已从旧实验版本目录迁出。

当前实际覆盖、计时与资源终态见[本轮结果](../../.pipeline/experiments/operator-consolidation-20261019/RESULT.md)，旧路径到新位置见[迁移表](../../.pipeline/experiments/operator-consolidation-20261019/PATH-MAPPING.md)。

重构前完整源码讲解保存在Git提交e6f1ba51902227c057a0a7ca2f72934d900769c9；旧数值与失败记录保持原身份。投机侧源码/配置/评估入口和共享依赖未修改。
