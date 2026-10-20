# 当前context128成本分析

同M39有界分析；[登记](REGISTRATION.json)是用户新请求下的新阶段，不重写旧1497/1500、372/420、20:20:28历史截止。

- [ ] 从已验当前源码生成隔离stage-clock插桩，生产optimized/二进制不变。
- [ ] 三真实B14 KV，各native与3次完整插桩block，逐位检查中间量/有效KV；合计12block。
- [ ] 同一真实阶段输入对照Torch Linear，最多150原语调用；只作归因，不混入正式速度。
- [ ] 若现有NCU权限/工具可用，最多三次当前未插桩kernel采计数，内部replay≤32；不足则UNKNOWN。
- [ ] 汇总阶段/寄存器/流量证据、线程与DAG源码讲解、优化排序；资源终态和HW。

插桩时间包含阶段边界等待，不能等同纯原语或直接分离barrier成本；profile输出须与未改数学对照一致。无新shape网格、无训练/机器人集成。

21:40适应性细化：NCU指标查询返回ERR_NVGPUCTRPERM（未启动profile kernel，不改权限/驱动），硬件counter保留UNKNOWN。利用原24block帽中剩余12次，三KV各一次未插桩当前block和三次native完整block profile，按kernel名/实际grid区分cross-K RMS、table、GELU等，输出/有效KV再配对。原语150帽不变，不追加tile或数学改动。
