# 冻结TF_ON的Long完整三臂评估

独立研究入口，不覆盖4f04科学快照、9acc旧PF2源码或旧NO_GO。Long10×50官方初态×T/C/P共1500完整episode，双A100-PCIe各一个持久target。同case同卡，init-major均衡分片，三臂顺序轮换；T原canonical CFG2 full50/terminal，C连续投机，P=SST。

真实作业：`投机采样/results/runtime/p0-tf-fullsuite-long-v1-20261017-r1/plan.json`。新48h生命周期，含等待、失败、清理；单局1200秒/4000forward，全作业6001000forward，10GiB含3GiB代表fixture。仅从source/资产实际就绪后运行launch.py，源码冻结身份来自单独Git提交；所有旧四权限false。原资产与完整512-token BF16优化reset bank复用，所有对照相同。

用户明确不做smoke、独立测试或SHA。必要审计随完整任务记录；失败/超时保留，不自动重试。自然失败与raw隔离差异不停止描述覆盖，但隔离差异令严格pair INVALID；非法路由、API、数学/资源合同仍停止新工作。全部50初态包含历史使用组，不能将其称独立TEST或非劣证明。

质量并行时只保存三份真实request前状态/输入与小oracle，不复制verifier全KV；replay捕获为未验证材料，必须之后本机独占重放核验边界与完整有效状态。原捕获tensor是CPU连续副本，原shape/stride另列，不声称已经恢复原物理布局。捕获/审计开销属于探索运行，正式性能排除。完整chunk replay不能替代完整episode E2E。

`launch.py --job <job>`只启动该注册runner与独立绝对guard。`runner.py`执行完整case；`report.py --job <job>`在CPU生成描述质量、长度、800罚分、有效提交成本及缺口，不能更新原质量权限。所有owned进程用PID、出生ticks、PGID审计；进程退出不等于实例关机。

LIBERO90另需已核实同代target与独立draft；本入口明确只支持Long，不盲用Long/FastWAM冒充90。HW导航：`.pipeline/experiments/libero-fullscale-20261017/PLAN.md`。
