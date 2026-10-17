---
kind: decision
name: 旧AutoDL阶段收尾暂停并准备换机
level: constraint
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-17T21:12:00+08:00'
supersedes:
  - '旧AutoDL实例在Long结束后自动接续Goal/Spatial/缓存轮换；现改为阶段收尾后暂停待换机'
sources:
  - '用户：autodl大概还需要多久，我等会儿给你换一个新的服务器可以嘛，你可以等阶段性的出结果就把autodl的程序暂停吗'
---

# 旧AutoDL完成当前已进入配对后暂停，保留证据并准备迁移

用户允许将研究换到随后提供的新服务器，并要求现有AutoDL取得阶段结果后暂停。此决定只改变当前实例的调度，不取消四套研究范围，也不授权自动购买新机器或重启失败作业。

当前Long先收尾已开始的完整Target/TF/TF+PF2三臂组，不截断正常episode，不按成败选择暂停边界；阻止新case和后续Goal/Spatial模型、训练及缓存轮换自动开工。采用独立有界调度入口，校验自有PID/出生/UID，仅控制本项目进程。旧数学源码、FAILED、预算起点、绝对截止、阈值、retention及四项false不改。

阶段结果、完整与未完成清单、选定draft/normalizer、必要bank、配置和原始证据须可追溯保存。迁移按完整case去重，同一新case的各臂仍固定同卡，硬件和并发阶段分层，不混作正式计时；不把阶段面板称全量完成或独立TEST。

停止GPU任务、CPU等待/下载/归档控制器、完成备份和关闭租卡实例分别核验。未确认必要证据安全前，不宣称旧机可销毁。新服务器连接信息尚未提供，不能声称已迁移或恢复；换机不重置原实验预算。
