# 重放入口与公开遗漏

保留L13源码位于仓库算子研究/wam_ops/optimized/csrc/whole_layer.cu，固定提交1ef5b83af38d523b835f2511018334ba99a4b694。A固定319ac6a；两者独立二进制路径/hash见SOURCE-MANIFEST。

make_bc.py/make_ln.py/make_diagnostics.py是本次原始生成脚本，运行时worktree为A。现在本地worktree已提交L13，不能直接在原实验目录重跑这些脚本；重放需新namespace＋A锚点工作树，保留原attempt。CHANGE.patch和variants/*/whole_layer.cu保存实际受测源码。run_job.py拒绝覆盖已有STATUS；新作业仍需有效预算与资源窗口。

公开快照包含源码、差异、原始JSON/CSV、审计、失败及报告；不包含原模型、真实PT输入、二进制、编译缓存。不能仅凭GitHub声称重算全部原tensor。实际fixture、checkpoint加载函数、框架版本与固定源码身份见manifest/每个JOB/worker。原whole-model hash禁令遵守，未运行整模型hash。

T2-diagnostic首失败涉及allocated logical_slots无效尾部；r2保存raw differences与完整有效范围比较，不降低数值标准。B/C控制不纯见REVIEW_RETURN。
