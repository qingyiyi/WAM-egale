# FFN2 split-K 有界测量入口

仅准备CPU脚本；本文件不代表GPU已运行或结果已取得。不修改原Torch、A、生产投机或旧失败。正常候选由主执行者提供。

- A独立二进制：本目录 `A.so`，加载名 `wam_whole_grain_cuda`。
- 正常候选：`wam_whole_splitk_cuda`，构建路径由 `--candidate` 指定。
- 诊断候选：`wam_whole_splitk_profile_cuda`，17区间globaltimer；不能用于正式整层速度结论。
- 必要登记：本目录 `JOB.json`，须包含真实 `absolute_cleanup_deadline`。旧截止/账本不得覆盖或重置。
- 测量输入：既有 `tools.block_loader.FIXTURES` 的真实B14/Q16/block0短/中/长KV三文件；不生成随机输入冒充真实fixture。
- 四臂：原Torch、Guarded、冻结A、本候选，全部eager、同等预计算Cross K/V，无CUDA Graph。

## 执行顺序

1. 候选正常源码与唯一二进制固定后，运行 `precision_unique.py --binary … --module-name wam_whole_splitk_cuda --output results/SPLITK-precision-data`。
2. 数值、缺字段、输入、有效KV、RNG均有限通过才计时。脚本对数值FAIL返回非零，不只对异常返回非零。
3. 正常候选源码固定后运行 `make_profile.py`；它仅生成本目录 `whole_layer_profile.cu` 与 `build_whole_profile.py`，不会覆盖optimized。
4. 两个独立进程顺序执行 `measure.py`，分别 `--process 1` / `2`。共同参数 `--candidate … --module-name wam_whole_splitk_cuda --label SPLITK`。只给第一个进程 `--stages --profile …`；第二个只正常测量。输出使用 `results/SPLITK-measure1-data` / `…2-data`。
5. 核对 `STAGE-DEFINITIONS.json` 与冻结核心真实实现，填入实际同步/任务/warp分工后运行 `report.py SPLITK --retained '实际当前保留版本'`。程序保存并直接打印完整17行表和各独立进程结果，不自动决定采用。

上述GPU命令须由主执行者在登记与授权窗口内运行；CPU准备未运行这些命令。

## 调用预算

每个测量进程硬上限6500完整block调用，两个进程最多13000；精度脚本3×3=9完整block。实际常规测量约7400完整block，另有独立Torch阶段调用，后者不冒称完整block，但其生命周期消耗仍计预算。主执行者累计本轮所有失败、重复、准备/清理；最多15000完整block，不因每进程上限而扩张总额。

每个KV：30轮四臂预热，24轮交错×10完整block；第一进程附15轮正常/插桩预热和24×5正常/插桩完整block区间。保留每一轮时间、同轮A与候选、同轮Torch与候选。原Torch独立阶段16×20测量只用于诊断，不把阶段总和代替完整层。

`run_stage.py`沿用owned PID＋出生时间检查、进程组清理和真实退出状态；自身不重试，阶段名与目录必须唯一。正常测量最长180秒、构建最长900秒，且受总deadline减90秒清理余量约束。

## 独立FFN2的最小接口方案（尚未实现）

现有pybind只有完整 `forward`，仅从完整FFN区间无法把FFN2算术、队列/提交和FFN1等待分开。要独立计时，需在同一候选源暴露一个诊断入口，例如 `ffn2_replay(act, weight, bias, residual, temb, table, debug)`：

1. `act`来自真实原Torch capture的GELU输出 `[224,14336]`，原dtype BF16；其余使用同一真实fixture的权重、R2、temb及table。
2. 绑定同一Params必需字段，设置FFN1 ready完成，调用与整层完全相同的FFN2 device函数和按序BF16提交；独立kernel使用同一CTA数、Shared存储和线程数。不要另写“近似FFN2”。
3. 对照必须包含同样epilogue：`(residual.float() + block.ffn.net[2](act).float()*cgate).to(BF16)`，该索引已沿真实capture工具与模型构造核实。纯Linear另列，不能与融合残差入口混比。
4. 预分配输出/队列等workspace，让正常独立路径仅发射kernel；如整层绑定含分配/host开销，完整层仍按原边界测量。
5. 所有27输出/分段舍入、写回与状态检查保持；独立输入复用不代表完整chunk或机器人质量。

若此接口会增加一次完整大kernel编译，当前轮可先保留UNKNOWN：优先本轮同场完整层与融合FFN区域的因果结论，不为补独立指标突破预算。
