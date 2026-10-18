---
kind: experiment
name: 原始与verifier规模短中长KV性能精度对照
cycles: [wam-research-20261016]
status: complete_evidence_with_standalone_full30_failure
created: '2026-10-18'
---

# 用户新增授权的算子真实输入矩阵

**当前终态（20:44）：** [性能、精度与完整输入输出统一结果](COMBINED-RESULT.md)已齐。r6新54点性能协议完成；后续用户批准共享精度，r8的54点/24完整模型执行、r9九保存输入27比较、24输出CPU核对均已完成。standalone完整30层3/6未过旧容差，shared/guarded有限六query逐位及有效状态通过。全部自有进程已退出，生命周期1649.274/1800秒。性能资源异常与block波动未消除，仍探索性。下方原计划及带时刻的过程记录仅属历史；原双进程7轮协议31点PARTIAL未改。

用户2026-10-18明确允许在本机补齐算子缺测，要求原始输入与verify输入规模各短/中/长KV性能和精度。当前步骤：真实输入/版本核查→本机质量任务自然收尾→独占测量→全部结果归档。

- 原始action：CFG2/B2、Q16；原始video Q128按真实输入可得性单列，不能用截取历史伪造对应query。
- verifier：7候选×CFG2=B14，每候选Q16；保留真实公共历史、私有KV、CFG分组和mask，不把候选拼成Q112序列。
- 目标KV：288/1152/2160；用真实fixture实际长度核对。既有旧六输入的K128/144结果保留，不能改标签。
- 输入首先核验 `p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4/{short,middle,long}/real-verifier-fixture.pt`。第一/最后query及层0/8/29优先；不新增机器人采样，不生成随机输入。
- baseline：同版本Torch SDPA/实际最佳shared-prefix路径；原standalone SDPA与后续G1/G2/guarded独立列，不继承旧证书。当前阶段只测真实算子/必要block机制，非正式45chunk或机器人E2E。
- 数值：保存max_abs、rel_L2、torch.equal/原始字节比较、元素误差分布和原allclose结果；不放宽旧标准，不以容差通过替代接受路径/全KV/RNG证明。
- 性能：eager与Graph-core分列，布局/pack/必要输出成本单列并给包含版本；两独立进程、每点7轮交错，保留全部慢轮/原样本。profile不进正式时间表。
- 资源：CPU准备≤2核/60分钟；GPU测量生命周期合计≤30分钟，单worker、含5分钟清理；独占等待上限60分钟，整体墙钟≤2小时；新增证据≤512MiB，不复制既有2.8GiB完整fixture/大模型。超时/准备失败计账且输出PARTIAL，不能自动选最好轮重跑。
- 不改原tile代码/HW、原数学实现、state/cycle/rules或运行中的远端任务；独立namespace保存新入口和完整版本/输入manifest，不做SHA或smoke。
- 若源版本、布局或fixture字段不足，先给实际缺项；不声称未测输入通过。本机存在他人GPU进程时不测正式性能、不杀他人任务。

当前尚未启动本矩阵GPU测量。结果终态、原始数据、失败原因、自有PID出生时间及清理在对应runtime登记。

19:10更新：所有20份canonical输入/中档prefix原件与三档真实B14完整fixture均存在。独立入口 `投机采样/experiments/operator_kv_matrix_v1/` 已完成源码准备；canonical B2完整T policy replay抓video/action first/last，B14用真实冻结query；层0/8/29，计划54点/进程，两进程。旧native与Torch、B14强shared-prefix及G1/guard独立列。包含所有raw、元素误差分位数/BF16 ULP、实际block与有效KV核验；不写原tile。

本机Object已自然16条后CLEAN，剩14条迁往新机；但lhw于18:57启动PID1911432的newton_atomic GPU作业，19:00占约1.1GiB，正式计时未开始。已向用户询问20–30分钟独占窗口。CPU-only等待器每30秒核资源，最多1小时，无GPU初始化；资源实际空闲后仅启动一次预登记矩阵。等待与实际GPU生命周期分列并纳入总体2小时，GPU测试/清理合计上限30分钟。[runtime](/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-kv-matrix-local-v1-20261018-r1)。

19:24更新：用户已明确先等lhw自然结束。r1仅CLI入口失败（0forward，42.003秒），r2资源未准入（0forward，82.136秒），原PARTIAL保留并累计扣124.138秒。r3 CPU-only等待最多1800秒、连续空闲120秒后仅启动一次；测量生命周期剩1675.862秒，不重置原30分钟帽。不把资源采样/等待写作测量完成。[当前runtime](/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-kv-matrix-local-v1-20261018-r3)。

19:42更新：lhw后续GPU作业仍在运行，r3尚未释放，GPU_jobs_started=0。CPU返回器1938389/birth1359278650已实际启动，单核、最多3600秒，只等r3终态并聚合所有原始样本；不启动/重试GPU，不改旧门。成功、等待超时或测量失败均输出真实报告，PARTIAL进程的时间只作诊断。原总2小时墙钟不延长，新增派生证据≤32MiB且计512MiB总体帽。[机器可读当前状态](CURRENT-STATUS.json)。

19:51更新：用户新明确持续监控、空闲即启动，并确认现在可采集。r3 CPU等待已自然终态BLOCKED，实际1831.139秒（包含最后采样/睡眠收尾），0GPU；原记录保留。当前可见GPU进程为空、显存0MiB、功耗32W，但NVML util21–25%根因UNKNOWN。r4独立资源入口允许稳定空context采集，原始读数不变、计时窗口资格false、延迟仅探索性；不是放宽数值或质量门。worker1950378/birth1359332989已实际启动，数学worker未改；保留原剩1675.862秒和总体21:00截止。目标54点/进程×2，原始样本/失败全部保存。

19:58更新：r4因wrapper遗漏canonical reset初始化，原snapshot.restore字段缺失，0forward终态PARTIAL、清理已观察；126.311秒计原预算。独立worker_r5仅增加原reset(prompt=None)，无prompt编码/teacher，随后恢复全部记录状态/RNG；原执行源存r4/EXECUTED-worker-source.py。累计扣250.449秒、剩1549.551秒。r5实际完成240forward/54捕获点、6冻结B14完整oracle均逐位，第一进程9/54已测；第二进程尚未开始。未改旧数值/质量门，延迟仍探索性。

20:13更新：用户新明确剩余独占不足20分钟，性能优先、I/O后验。r5按现有检查点收尾31完整点、240forward，lifecycle774.179秒，原PARTIAL保留，不把单进程冒充双进程完成。新r6独立性能协议54点、单进程3轮，attention每轮10样本、Graph8调用、block每轮5样本；I/O/有效KV明确PENDING。worker1961771/birth1359449234实际运行，35/54点已测，9代表输入370763331B已保存。原剩775.372秒且绝对20:19释放/提前60秒清理；不动旧质量门和tile。
