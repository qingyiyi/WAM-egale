# 数据与版本说明

本包公开标量、统计表、协议、报告、图表与投机源码，支持核对数字及重算统计；模型/预测器权重、teacher方向张量、相机/完整动作/视觉KV、DSO留在三台原机器，不公开。它不是一份下载即可重放机器人实验的数据集。

|入口|内容与分母|
|---|---|
|[episodes-6000.csv](data/episodes-6000.csv)|四suite×10任务×50初态×3臂的完整逐局成败、真实命令数、policy chunk、初态/seed/本轮数据暴露；由原QUALITY JSON的raw_audits和paired rows派生，非原EPISODE原文|
|[tasks-120.csv](data/tasks-120.csv)|40任务×3臂，含N、成功数、SR、平均真实命令；task为固定suite内官方索引|
|[paired-quality-4000.csv](data/paired-quality-4000.csv)|每个初态分别T–S/T–P配对，包含全部成败，不只成功子集；物理/身份/像素exact分列|
|[draft-selection.json](data/draft-selection.json)|每suite固定final3000、SHA、tau、Q80、query数、训练时间与dtype|
|[teacher-collection-summary.json](data/teacher-collection-summary.json)|每suite100FIT+10CAL；FIT方向行数、CAL保存行数、用于阈值的行数；原DATA-MANIFEST逐episode保留|
|[calibration-scores.csv](data/calibration-scores.csv)|4960个用于阈值的CAL双头最小分数及scheduler索引，来自冻结CAL-SCORES；不包含teacher向量；按suite重算线性Q80可得到tau|
|[cached-speed-pairs-1125.csv](data/cached-speed-pairs-1125.csv)|共同prompt缓存版完整chunk与独立action测量，225观测×5，原DEV三臂共同成功来源75组；包含全部快慢调用和实际target/draft/提交计数|
|[cached-profile-phases.csv](data/cached-profile-phases.csv)|每臂75次独立同步profile；full_policy/action_denoise为嵌套边界，不能把所有行相加|
|[robotwin-canonical-4.csv](data/robotwin-canonical-4.csv)|4任务工程canonical，实际有效scene seed、成败、命令/chunk及并行耗时；不是速度benchmark|
|[full-eval-resource-jobs.csv](data/full-eval-resource-jobs.csv)|12个全量作业真实北京时间起止及生命周期；不包含前置采集/训练等总账|

原始JSON保持在仓库`.pipeline/experiments/`相同相对路径；以下是本轮核心原件：

- 四suite各自`fastwam-full-triarm-20261027/<suite>/QUALITY-FULL-TRIARM.json`：1500逐局审计、500组T–S及500组T–P配对、逐任务和全体成败；`RESULT-REVIEW-v1.json`另含区间；Object v2 Markdown仅纠正初始像素文案。
- `fastwam-full-triarm-20261027/TEXT-CACHE-v1/<suite>/results/speed-triarm-attempt01/`：`FULL-CHUNKS.jsonl`、`ACTION-CHUNKS.jsonl`、`POLICY-TRACES.jsonl`、`PROFILE-CHUNKS.jsonl`、缓存等价性/批验证/数值敏感性/模型加载/配置/结果JSON。
- 原未缓存速度也保留：Long在`fastwam-base-small-closedloop-20261026/SPEED-5090-archive/speed-triarm-attempt01/`，其余在各suite `SPEED-5090/results/speed-triarm-attempt01/`。这组原计时含失败来源，只有`SPEED-COMMON-SUCCESS-v1.json`按用户口径筛选为速度主表；不要拿原RESULTS的全输入倍数替代。
- Robo `CANONICAL-REVIEW-v1.json`是后续技术+人工图像审阅结论；之前的`attempt04/review-v1/TECHNICAL-REVIEW.json`仍保留WAITING_AGENT_VISUAL_REVIEW状态及原SHA，未事后篡改。
- 算子`fastwam-crosskv-cache-20261027/`包含最后报告、层/完整policy原计时与数值记录；与主投机速度不是同一实验。

T=canonical，S=TF真实投机，P=TF+PF的S,S,T周期。`commands`为实际环境动作命令，`chunks`为完整policy调用次数；每chunk都完成10次scheduler更新。所有失败仍进入成功率及全体长度。配对trace/速度仅在指定共同成功集合统计，不能当总体无条件收益。

public CSV中的`complete=1`和success取自原审计/配对数据；细分failure原因以私有原EPISODE终态为准，不能从这个CSV杜撰超时/碰撞类别。当前原报告已确认缺失/PARTIAL/无效均0。`historically_exposed_panel`不等于独立TEST，其余初态没有用于本次配方选择，也不能清除历史暴露。

[MANIFEST.json](MANIFEST.json)保存固定代码/权重身份、每个来源文件的原SHA与公开SHA。正文与数据中的私有执行路径被占位符替换，凡发生改写均显式标记，科学值不变；SHA对比应使用对应的public_sha256。JSON原件可能仍引用未公开大tensor或历史附件，这是保留来源定位，不表示附件已公开。新文件完整清单与SHA见[PUBLICATION-MANIFEST.json](PUBLICATION-MANIFEST.json)。

独立CPU核验入口：[verify_publication.py](verify_publication.py)。在仓库根运行`python review/fastwam-20261027/verify_publication.py`，核对全部导出SHA、6000逐局汇总/120任务行、1125配对与每chunk十步；不访问GPU或远端机器。
