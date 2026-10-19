# 最近执行

## 2026-10-20T01:34:15.011424+08:00：M34远端首批闭合、M35未启动分片接续

远端9完整block trace单kernel成立；原数值门FAIL，270原始计时样本中新kernel比冻结native慢5.12–11.28倍，450 block尝试，3GPU roots已reap/context释放。83调用有界profile/单共享布局挑战者另登记，主源码/DSO冻结。M35 Object1误送Goal/Spatial专用入口，0局0teacher失败保留不重跑；只接续原未启动Object2/3，原360缺30不算算法失败，预算/截止不变。Goal/Spatial90各齐，Long原自动队列已运行。

## 2026-10-20T00:49:56.738555+08:00：M34整层结构实测与数值FAIL

本机空闲窗口完成9真实block重放，2 trace各1整层kernel、tile/输入/RNG过；9输出逐位失败。只读KV9过，写序列18中9数值失败/metadata全过，未集成。原默认Python0block依赖失败保留。cuBLASLt公共API元数据核对B2 FFN2 splitK7/B14 splitK2；172寄存器使理论1CTA/SM。原local三个worker已退出；远端GPU算子仍CPU等空卡。M35 Goal90齐，总156/360（00:44），无重试或四false变更。

## 2026-10-20T00:07:07.268346+08:00：M34现在实施

用户最新要求算子侧直接实现、明早汇报。CPU编码/编译与M35并行；算子GPU验证只用计时队列已结束并清理的卡。保持单层、全部tile单kernel、warp职责分工和原精度门；未把实现或验收写成已完成。

## 2026-10-19T23:45:01.681439+08:00：M35新独占工程计时

四套原6000局纯CPU核对完成，Object1500逐终态/COUNTER重新读取。GPU0 Goal和GPU1 Spatial实际READY并已有完整episode；各自单worker，Object与Long排后继。新360局登记与旧6000质量N分列，四false/FAILED/旧预算不变。旧45chunk入口未启动。本轮远端工程计时而非旧质量证书放行。

## 2026-10-19T22:56:41.237773+08:00：M33收口

[M33实际profile](../../experiments/single-block-persistent-profile-20261019/RESULT.md)完成。B14长KV静态/阶段内persistent均34 kernel，增量207.072us由self/cross attention与FFN1合计207.2425us解释。108 CTA上限低于理论2 CTA/SM容量；隔离216 CTA反事实使B14短中长相对静态退化从4.154%/4.781%/5.459%降到0.726%/0.474%/0.179%，没有新增加速。不锁时钟，跨case绝对时间非单调，不能作正式性能。

九真实block×四模式两次profile（153调用）、有效三B14反事实112调用，输出/有效KV/RNG通过。NCU权限失败，实测occupancy/L2/DRAM UNKNOWN；两个扩展身份/符号实施失败保留，第三attempt隔离后完成。五GPU job生命周期81.856秒、293 block尝试含失败、0full30/0episode；22:53两GPU无进程，全部owned退出，新证据42.91MiB。baseline及依赖88项不变。

用户要求的是**整个单层统一persistent kernel，内部warp组调度全部tile**。当前M32只有attention/FFN1阶段内常驻循环，QKV/Wo/FFN2仍host cuBLASLt，不满足整层目标。[完整DAG与缺口](../../experiments/single-block-persistent-profile-20261019/WHOLE-LAYER-EXECUTOR-GAPS.md)已登记，整层实现仍待完成。下一M34补设备原语、ready/生命周期与统一执行器，不扩30层，不以新wrapper或多kernel包装宣称完成。

原M32与前执行见[收口前归档](history/M33-before-summary-20261019/EXECUTION.md)。未触碰保护state/rules、投机源码、原tile，未升级系统。
