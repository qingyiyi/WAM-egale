# 本机整层单kernel：结构成立，原数值门失败

0:38–0:39固定va/Torch2.9环境、真实Long block0权重、9个已有真实输入；只做完整单层重放，没有机器人episode或30层。原结果原样保留。

| 输入 | max abs | 相对L2 | 输出逐位 | 只读有效KV | 写KV序列通过 |
|---|---|---|---|---|---|
|short-B2-action|1|0.00237295|FAIL|PASS|0/3|
|middle-B2-action|1|0.0035992|FAIL|PASS|0/3|
|long-B2-action|1|0.00317303|FAIL|PASS|0/3|
|short-B14-action|1.5|0.00559703|FAIL|PASS|0/0|
|middle-B14-action|1.5|0.0034378|FAIL|PASS|0/0|
|long-B14-action|1.5|0.0042738|FAIL|PASS|0/0|
|short-B2-video|0.5|0.00130661|FAIL|PASS|3/3|
|middle-B2-video|0.5|0.0016981|FAIL|PASS|3/3|
|long-B2-video|0.5|0.00137187|FAIL|PASS|3/3|

两条short action B2/B14 profiler trace各只有`wam_whole_layer_v1::whole_layer`一次计算launch。九输入tile恰好一次、输入/RNG/参数地址与grad flag通过。只读9状态通过；18次合法写序列中video9次通过、action9次K/V数值不通过，metadata均通过。全部输出有限，但9/9输出逐位失败；不以小相对误差改门。

实际函数资源：256线程/CTA，172寄存器/thread，59616B动态shared，LOCAL0，occupancy API给出理论1 CTA/SM、108 CTA。该值不是实测achieved occupancy。全层仍global workspace和阶段grid barrier，没有跨阶段流水。

相同输入控制：Q投影已有小差异，FFN2最大绝对差0.25。全层差异同时包括GEMM舍入与attention顺序，不能只归因某一个。旧M26四公式失败保留，未重复猜测扫参。

原local-correctness-2中的`first_different_intermediate`按map字母顺序产生，不代表首个DAG分歧；本派生CSV按完整已保存中间量metrics重新排序，未改原字段或声称重新算张量。首个Q投影same-input控制单列。

原Torch与冻结native对照、历史gold、逐项误差、状态、trace、owned清理见`local-correctness-2/results/RESULT.json`与`OWNED-RESULT.json`。完整记录65 block尝试＋18个独立Linear控制；CUDA active秒UNKNOWN。

首次local-correctness-1误用默认Python/Torch2.6，在缺easydict处退出，0block/无模型GPU调用，6.255秒生命周期；已保留，未补PASS。第二次显式固定va解释器成功完成结构研究，26.077秒生命周期。

后续公有cuBLASLt API元数据读取完成：2真实block、8 heuristic query、0新增GEMM执行（查询本身）。B2 FFN2 splitK7/inplace，B14 Q与FFN2 splitK2/inplace；原workspace preference1MiB，BF16 reduced-precision reduction允许。新WMMA为单段FP32累加，契约与原库算法不同。该metadata为匹配descriptor的API重建，不是已结束调用的直接拦截；不单凭配置差异断言根因。

本机阶段已退出；不集成生产，不改旧四false。远端还将按新登记验证/计时，结果不与本机或旧M32混写。
