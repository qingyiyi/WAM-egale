---
kind: preference
name: Long重复评估汇报三轮均值与波动
level: guideline
scope: project:WAM-egale;Long评估设计
status: active
updated: '2026-10-18T01:25:24.688361+08:00'
supersedes: []
sources:
  - '用户：想跑三轮，每轮Long10的10task×50init，三次取平均，询问是否合适'
---

# Long重复评估倾向三轮并呈现波动

用户希望每轮覆盖同一10任务×50初态，共三轮后报告平均成功率。实施建议是事前登记三个评估seed，轮内Target/TF/PF2相同seed配对，轮间改变seed；逐轮结果、均值与标准差一并报告，不能只留平均。重复初态不变，因此不能宣称1500独立初态或独立TEST；固定相同seed的重复主要衡量复现性。

这条偏好不视为已冻结后两轮seed、修改当前20260922执行协议或追加GPU/forward预算的授权。先完成当前首轮，在新namespace登记后两轮范围与预算，再执行。Long三方法三轮总4500完整episode，不能将每方法1500与总数混淆。
