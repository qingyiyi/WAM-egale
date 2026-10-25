# 资源与运行

先读[现行约束](../CURRENT-CONSTRAINTS.md)。以下按任务选用，不自动逐项全读；当前数字读Cycle。

- [本机固定Goal接续](../global/requirements/decision-本机Goal失败先修复再有界接续.md)
- [双A100与5090分工](../global/requirements/decision-A100双卡专用于RoboTwin算子与性能统一5090.md)
- [RoboTwin 160GPU小时原预算](../global/requirements/decision-RoboTwin扩大评估授权160GPU小时双卡接续.md)
- [现有5090持续授权](../global/requirements/decision-允许持续使用现有5090开展算子研究.md)
- [后台巡检与实际终态](../global/requirements/requirement-任务完成前定期核验服务器运行与进度.md)
- [前台算子与后台质量](../global/requirements/preference-后台质量评估接续且算子汇报附三张A100简报.md)
- [分机与本机轻量化](../global/requirements/decision-成功率与性能分机执行并将本机改为轻量工作区.md)
- [临时目录](../global/requirements/requirement-Codex临时文件统一使用账户tmp目录.md)

实际job计划/status优先：旧10/16、10/17、09:00、15:45等窗口只是各自历史。M40原160GPU小时/10天不因接续重置；M42新窗口独立登记。平台uid0不代表宿主管理权限；清理只处理出生身份匹配的owned进程。
