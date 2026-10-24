---
kind: decision
name: 训练验证与LIBERO标准评测的区别
level: reference
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-18'
supersedes: []
sources:
  - '用户询问是否未划分训练/测试，以及其他论文如何测成功率'
  - 'OpenVLA 2406.09246v3 Appendix E.1'
  - 'LingBot-VA 2601.21998v1 Section4.3.2'
  - 'FLASH 2605.13778v1 Section4.1及官方代码da6ceccad603695a8a3d6fa14dd410c3aadb536f'
  - '.pipeline/experiments/long-morning-20261018/TRAINING-COVERAGE-ANALYSIS.md'
---

# 有训练与校准分区，不等于最终面板全部独立

当前有按episode隔离的FIT/CAL及用于选方案的DEV。Long原FIT174/CAL20与后续FIT20(init30/31)/CAL10(init32)不是随机混拆query行；但最终10任务×50初态包含这些与旧DEV来源，所以全500不是从未参与训练和选择的独立TEST。完整成败统计仍保留，不把曝光问题说成500局全部无效，也不事后把“非FIT”改名独立测试。

已核论文常在同一LIBERO任务套件的离线人类演示上训练，再做新闭环rollout；不要求训练/测试任务名称完全不同。OpenVLA与LingBot-VA报告每套500局、三个随机seed；FLASH报告每任务50局，并分别训练suite draft。所读原文没有提供离线演示与官方init文件的逐项对应表，不能声称这些论文均保证初态完全不重叠，也不能断言它们使用了重叠初态。

FLASH固定代码的enc_cache从LeRobot演示生成teacher标签；spec_draft_train默认val_frac=0.1并按episode划分，离线验证误差不是闭环成功率。可借鉴其数据组织，但它预测action chunk，我们预测连续去噪velocity，不能直接继承训练效果。

标准50初态完整面板、独立保留集、跨任务/OOD泛化是不同主张。当前建议保留标准面板与数据暴露说明，并在后续协议中前瞻规定保留集及选择规则；建议不是新TEST执行或修改旧门的授权。三个新套件原配方与分区不变，未用于其训练的init26–49仍只称未参与该训练的子集，不自动签独立TEST。换seed重复不自动变为新独立初态。

详见[原文核查和建议](../../../experiments/long-morning-20261018/EVALUATION-PROTOCOL-LITERATURE-20261018.md)。四项false、旧FAILED与原质量门不因其他论文的做法而改写。
