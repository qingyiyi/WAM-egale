# 当前入口

请先读[2026-10-19当前结果与源码](CURRENT-20261019.md)。M30已补真实shape/DAG与MegaMoE思想设计，未实现新persistent；本次完整native block的有限逐位与两进程动态计时已完成；最强B14/Graph与闭环仍未验，四质量权限仍false。main只作进度镜像，固定科学commit保留身份。

以下正文与review/RESULTS.json为4f04 paused72历史快照，不能覆盖后续新授权运行或当前算子结果；旧FAILED/retention亦不能由新进度改写。

---

# 新版TF draft审阅入口：暂停72局

当前snapshot_id：`wam-tf-draft-handoff-20261017-paused72`。固定本文件所在的新提交，旧科学基线`b0dfac3a02f65de33a1217246b83003615166640`只用于历史；不要随main更新。

用户已暂停GPU，先请网页版查原文与官方开源，为TF结构/训练给出一条有限且可证伪主线。暂停不是新的训练或TEST授权。

## 最新结果

- 一次30完整新数据、TF-cont/TF-on/MLP-on公平500更新完成。数据来自旧TF_CONT3000，不是新TF-on自己的on-policy。
- 固定96完成72条机器人终态，24未完成；50条逐局必要组件审计完整，22旧TF-on arm清理收据缺失，总体严格证书PARTIAL。全局清理证据不补造局部收据；暂停不是预算completion，原issue保留。
- fresh20：T20/20、CONT19/20、TF-on20/20；MLP-on0/20。TF-on/T共同成功步数P50/P90/max=.990745/1.020326/1.027027，仅描述，不签严格PASS。
- TF-on风险已完整4/8，0/4成功；task1/29、3/26、4/26 T成功/TF失败，task5/28双方失败。三条新增失败均有初始物理/reset/RNG/双相机exact证据。当前配方不应推荐独立TEST；TF架构不是已证伪。
- 四项权限仍false；旧A_off learned draft默认未改。120、独立TEST、新TF正式45chunk/新系统实验未启动，ΣT_round/Σg和新TF正式E2E UNKNOWN。

## 先读这些

1. [完整版文献/代码调研Prompt](TF_DRAFT_RESEARCH_PROMPT.md)
2. [结果JSON](RESULTS.json)、[本轮总览与五报告](../.pipeline/experiments/tf-effect-closure-20261016/RESULT.md)、[短HW](../.pipeline/PROGRESS.compact.md)
3. [TF/MLP结构](../投机采样/experiments/draft_architecture_v1/models.py)、[固定训练](../投机采样/experiments/tf_effect_train_v1/train.py)、[CPU恢复与复用](../投机采样/experiments/tf_effect_train_cpu_reuse_v2/train.py)、[teacher采集](../投机采样/experiments/tf_effect_collector_v1/collector.py)
4. [接受和合法提交](../投机采样/scripts/p0_batched_verify.py)、[真实target](../投机采样/wam-p0-worktree/wan_va/modules/model.py)
5. [单文件阅读包](REVIEW_PACK.md)、[真实登记的有界派生摘要](REGISTERED_EXPERIMENT_SUMMARY.json)、[增量来源manifest](SOURCE_MANIFEST.json)、[遗漏原件](OMISSIONS-20261017.md)。

原始PT/NPY/相机/KV/模型权重不公开，不能声称仅凭GitHub已重算所有原始张量。服务器绝对路径是历史证据引用；在仓库中去掉`/rjs/cxz/论文探索/`前缀映射相对路径。
