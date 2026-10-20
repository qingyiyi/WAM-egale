# RoboTwin质量结果：尚无完整policy样本

|臂|计划DEV|实际完整DEV|成功/失败|SR、长度、罚分、E2E|
|---|---:|---:|---|---|
|Target|30|0|NA|UNKNOWN|
|TF|30|0|NA|UNKNOWN|
|TF+PF2|30|0|NA|UNKNOWN|

84局训练数据采集同样为0；未训练RoboTwin draft，未校准RoboTwin tau。新`robotwin_draft_v1`仅定义TF64接口/模型，未被称为已训练或已验收。

原Target首6局在原渲染前置受阻，详见[环境结果](ENVIRONMENT-STOP-RESULT.md)。专家准备不属于policy样本；基础设施停止不计模型算法失败。没有改变horizon、任务、seed、阈值或成功判据。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。原LIBERO四套结果不并入本表。
