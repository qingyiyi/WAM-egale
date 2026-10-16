# 来源与许可证说明

仓库原有根目录LICENSE保留。本次新增快照未改变第三方文件既有许可证或版权所有者。

- `投机采样/wam-p0-worktree/`来自[Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va)，本地基础提交为`92a633fef31d7a7868c5848a0d358a0b43aaaea4`。保留其`LICENSE.txt`、原文件声明与必要文档；本地研究修改涉及`wan_va/configs/va_libero_cfg.py`、`wan_va/modules/model.py`、`wan_va/wan_va_server.py`。
- `投机采样/experiments/shared_prefix_torch_v2/csrc/vendor/`保留导出范围中的CUTLASS、FlashAttention、PyTorch相关源文件及对应原LICENSE/notice。阅读各子目录许可，不将这些上游实现归为本项目新贡献。
- 研究实验、分析与HW文件来自本地项目快照；文档本地链接作了相对路径转换。具体逐文件来源、原始/导出摘要和任何脱敏在`review/SOURCE_MANIFEST.json`记录。

未发布模型权重或原始机器人数据；源代码入口中出现外部数据路径，不代表这些数据随仓库分发。
