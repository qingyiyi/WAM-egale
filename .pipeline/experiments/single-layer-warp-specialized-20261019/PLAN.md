# M34：整层单kernel、tile与warp职责专门化

用户已明确要求：整个单层Transformer在一个persistent kernel内部运行，每个warp有对应职责，借鉴MegaMoE；最新用户已允许立即实施，CPU编码/编译与速度评估并行。当前M32/M33不满足此目标。baseline及实际依赖冻结，只有optimized新增独立实现/构建入口，投机源不动。

## 有界执行顺序

- [x] K0：固定真实单层DAG、接口和原语差异。以实际代码/shape划分任务和完整归约边界，既有Flash/CUTLASS逐位证据只属于原线程布局。
- [x] K1（实现与诊断完成，原逐位门FAIL）：补全部投影与FFN2设备原语；先用真实输入和原cuBLASLt比中间量。优先读取实际heuristic/资源/归约信息，不重复M26已失败四公式，不任意降低逐位门或无限调参。
- [x] K2（设备任务已实现，状态门分项记录）：Norm/调制/RoPE/残差/KV metadata与copy转成显式设备任务，保留FP32/BF16/FP64边界、合法private KV和错误传播。
- [x] K3（本机两trace确认整层单launch）：一个全层persistent kernel；明确warp角色、ready DAG、shared/global workspace与代际。所有计算在内完成，host只绑定和预分配；不在kernel中调用host cuBLASLt。
- [ ] K4：真实B2/B14 action及B2 video×三KV，原中间量/最终输出/全部有效KV/输入/参数/API/RNG；调度计数每tile恰好一次，检查producer先行与终止，无死锁；trace确认整层计算launch为1。
- [ ] K5：与冻结顺序baseline和适用最强cached/shared-prefix/Graph比较。报告全部case与重复；若只有结构完成而原精度门未过，明确UNADMITTED/PARTIAL，不替换生产，也不称完成所有验收。

## 第一版角色方案（待编译/验证，非既成结果）

每个常驻CTA先考虑8 warp：metadata调度1，activation加载1，weight/KV加载1，MMA/softmax/归约计算4，epilogue写回1。计算原语可由4个计算warp协作；各warp角色明确，不把单warp当成完整CTA原语调用者。具体数量是设计变量，用户没有指定8；SM80资源不可行则先交测量结论，再决定一种有依据的收缩，不做网格扫参。

shared pipeline每槽有task generation、operand-ready、compute-done、store-done/empty。加载warp只负责对应搬运、计算warp执行对应数学、写回warp完成global输出发布；metadata warp只能释放满足真实依赖的任务。首版固定不重叠global workspace，不立即用复杂ring降低内存。所有角色终止由同一invocation状态驱动。

现有Flash/CUTLASS的`threadIdx.x`/warp编号、CTA范围`__syncthreads`必须改为显式计算组坐标与组内同步；不能让计算warp组进入旧CTA barrier而其他角色不参与。SM80的`cp.async.wait_group`仅等待执行线程自己的复制，加载组完成复制并同步后才发布ready。不得假设其他warp执行wait即可保证producer完成。

关键DAG：全3072维Q/K RMSNorm等全部投影列；self非因果Q16等合法全部当前private KV；Wo等24个head；FFN2等14336维activation。只有ready任务进入执行；消费者不能占满CTA等未被调度的producer。普通shared memory不跨CTA共享。

## 资源与止损

CPU立即编码与编译；GPU只使用对应卡M35计时队列自然收口、核验owned退出后的空卡，再登记实际T0、停工作/清理绝对时间；不得干扰另一卡的计时。初始样机最多两个开发工作日、7 GPU生命周期小时，包含失败/启动/等待/清理；CPU编译最多2核；新证据/样机≤3GiB，不复制完整模型/全30层历史。最小device门失败时保留实际失败，先给明确差异，不用不存在的原语拼假整层。

SM80无TMA/WGMMA/UMMA/TMEM或动态warp寄存器重分配保证。统一kernel按最大资源占用可能使所有warp分配高寄存器，当前attention255 registers/thread尤其需要核验；CTA数按实际函数资源与理论驻留查询，不再把1 CTA/SM当普遍最优。

kernel完整性、精度与性能分开验收。K1若不能保持当前精度契约，研究实现可保留但不能集成；不能宣称整层已验证。正式主张仍需比强B14与有效提交成本，普通batch/cache收益不归新调度。

整层源码已运行；远端9 trace各1 kernel，原逐位与性能门FAIL。原型结果见RESULT.md，禁止集成。最终源码入口预计为optimized下独立whole-layer DSO，不覆盖baseline依赖或现行M32 DSO。

## 00:39本机实际结果

[9输入有限重放](LOCAL-CORRECTNESS.md)：结构/全tile/输入/RNG通过；9输出逐位FAIL，9只读KV通过，18写序列9通过/9数值失败，metadata均通过。保留默认Python依赖启动失败（0block）。同输入Linear控制与原cuBLASLt API metadata见结果。远端验证和探索性时间待空卡，未集成。

## 后续有限诊断

两独立进程确认新kernel相对native慢5.12–11.28倍。新增83 block调用/15分钟GPU子预算：先测29阶段PTX globaltimer，不访问NCU计数；只有三B14 GEMM合计≥50%且9输入逐位匹配未准入v1，才编译一次共享ldm32→40挑战者。不变主源码/DSO，不扫grid/register/padding，不降低旧数值门。见DIAGNOSTIC-REGISTRATION.json。
