# 整层原型：阶段归因和唯一布局诊断

原精度门仍为FAILED，原型未接入生产。此处“匹配”只指已失败v1的数学和状态隔离，不是匹配Torch。原RESULT.md/270计时样本保留。

9真实输入的profile和pad40均逐位匹配v1中间量、最终hidden与只读有效KV，输入/RNG不变；未追加合法写序列重验，不能把旧写入失败改成通过。两种诊断扩展的C++ namespace、模块身份和单kernel trace不同且经过核验。

首轮设备计时：三B14合计GEMM占74.66%，触发了事先登记的唯一ldm32→40 BF16共享布局实验。29阶段通过PTX globaltimer记录，包含阶段边界等待；不是指令级profile或实测硬件计数。

| 输入 | GEMM占比 | cross K/V投影占比 | self attention占比 | profile ms | pad40 ms | 诊断延迟下降 |
|---|---:|---:|---:|---:|---:|---:|
|short-B2-action|80.0%|40.5%|4.7%|7.1404|5.9034|17.3%|
|middle-B2-action|69.7%|35.6%|15.6%|7.9114|6.9228|12.5%|
|long-B2-action|60.7%|30.9%|25.4%|9.0281|8.0236|11.1%|
|short-B14-action|84.3%|63.1%|3.6%|23.3513|18.9153|19.0%|
|middle-B14-action|76.3%|57.2%|12.7%|24.1597|20.0146|17.2%|
|long-B14-action|65.7%|49.2%|24.7%|27.7064|23.7212|14.4%|
|short-B2-video|66.5%|21.8%|7.7%|10.5204|9.3763|10.9%|
|middle-B2-video|52.5%|17.1%|26.5%|13.3327|12.2330|8.2%|
|long-B2-video|42.3%|13.8%|40.2%|20.3010|18.9754|6.5%|

profile/pad40列是一独立进程、每格两次交错样本的CUDA stream中位数，只能支持本次布局的局部方向信号。设备内阶段表来自独立审计调用，不能把其绝对时间和另一调用的stream时间直接相减。未锁时钟，跨case绝对时间不作单调性结论；新增诊断没有给出与native同轮重测的改善幅度。

观察：B14大量时间花在cross K/V投影，长KV下self attention占比增加。已有静态prompt/cached路径就是强baseline的一部分；避免这些重复投影属于已有缓存收益，不能重新命名为MegaTransformer贡献。当前整层实现尚未接入该缓存协议。

主原型172 registers/thread；加计时profile为175，pad40为198。诊断两者shared均59616B、LOCAL0、理论1 CTA/SM；编译器资源也随源码变化，不能把这当成只改变bank conflict的硬件隔离。WMMA数学/调度代码保持，9输入match-v1；改善与共享布局敏感性一致，但没有NCU bank-conflict计数，不能断言唯一根因。仍是同步加载、粗阶段barrier和全局中间量，没有跨阶段生产者消费者细流水。单kernel和warp分工本身没有保证速度。

本阶段实际83完整block调用，M34累计600/600；0完整30层、0机器人episode。GPU预留生命周期86.720秒，其中两GPU worker生命周期28.622秒、CPU编译57.851秒；总CUDA active时间UNKNOWN。全部owned root已真实reap，GPU0 context为空。

止损：不继续padding/寄存器/grid搜索，不晋级新kernel，不扩30层。本轮实现与诊断收口。后续若继续，先解决匹配原GEMM/attention数值契约的可行设备原语，再在强cached B14上验证tile流水；不能靠放宽旧门或把已有cache记为新贡献获得PASS。
