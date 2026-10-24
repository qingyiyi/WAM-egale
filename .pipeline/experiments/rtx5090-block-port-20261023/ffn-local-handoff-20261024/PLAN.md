# FFN有界交接与驻留实验

授权：用户同意将已讨论方案实际实验；现有5090持续使用，旧账本不改。双A100继续RoboTwin，不占用。仅optimized研究路径，baseline/投机/protected不改。

- [x] B-INPUT：原672任务、四MMA和七K2048归并保持，W0专职TMA，八warp原提交，验证输入分工成本。
- [x] C-OVERLAP：同队列/算术；一TMA、三epilogue、四MMA，单结果槽与双输入槽跨任务重叠。
- [x] ACT-RESIDENT：有限原型或由实际能力/容量审查止损；不得把不可运行写成算法无效。
- [x] CTA-PAIR：核查实际whole_layer的cluster占用与可运行能力；支持才实现一组有依据的producer/consumer。
- [x] 逐候选完整阶段表、原Torch/Guarded/独立A两进程整层；采用者公开两流与回同步。

各GPU job先登记，B/C每个最多30分钟、4CPU、远端1.5GiB、本机回传每个≤150MiB；准备/编译/失败/清理计生命周期，保留90秒清理。完整block上限12000。先实际三KV有限逐位/状态/计数，通过才计时；全部eager，无Graph，原Torch/guarded/冻结SELF-QKV-TMA A。

当前保留SELF-QKV-TMA-4MMA。候选精度失败即不计时、保留证据；正确但无稳定整层收益则回退。ACT及双CTA不扩大成完整DAG/架构搜索。每项结束打印并保存完整17行表，未测标历史或未测。

输入：沿用三份真实B14/Q16/block0，KV288/1152/2160及原weights/cache/normalizer。四科学门及production_admitted=false。单层研究不等于在线verify/robot E2E。

B/C均有限精度通过，完整层两个进程均退化，保留当前独立A。详见各REVIEW_RETURN及STAGE-TABLE。

D修正版实际逐位通过，但整层2.03–2.26倍退化，不采用；初次编译失败保留。E双CTA实际实验继续。

终态CLOSED_RETAINED_A_NO_FFN_GAIN。四方案全部实际完成有界实验，全部保留负结果，保留A不变。
