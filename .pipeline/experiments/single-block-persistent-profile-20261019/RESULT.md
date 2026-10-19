# M33：单层 persistent 退化 profile

终态：PROFILE_COMPLETE_COUNTERS_UNAVAILABLE。2026-10-19，北京时间；实际GPU作业已退出。整层统一persistent kernel尚未实现，见[实现缺口与新目标](WHOLE-LAYER-EXECUTOR-GAPS.md)。

## OBSERVED

九真实block、四执行模式、每模式两次profile，共72条时间线；153次block调用。tiled/persistent分别启用或关闭FFN1融合；与原Torch和保存gold核对输出、全部有效KV和RNG。有效结果通过。数学源码与baseline依赖88项不变。

长KV、B14 action，同开FFN1融合时，两次profile的CUDA kernel时间均值如下。单位微秒；不是正式计时或E2E。

| 部分 | 静态tile | 108 CTA persistent | 增量 |
|---|---:|---:|---:|
| self attention | 421.296 | 541.1805 | 119.8845 |
| cross attention | 110.4115 | 146.7785 | 36.367 |
| FFN1+GELU | 185.0165 | 236.0075 | 50.991 |
| 全部GPU kernel之和 | 3267.3165 | 3474.3885 | 207.072 |

三个变更核合计增加207.2425微秒，与整层207.072微秒增量基本一致；其余核小幅波动。两臂均34次计算kernel launch，没有整层launch减少。GPU时间线间隙均值187.623→183.864微秒，并非本例退化来源。B2 action三KV的对应核解释约95.65%–98.58%增量；video没有启用FFN1融合，不能混称FFN1退化。

实际launch：B14 attention静态336 CTA、persistent108 CTA；FFN1静态448 CTA、persistent108 CTA。编译/运行时资源：attention均255 registers/thread、64KiB shared memory；FFN1为154→168 registers/thread、72KiB shared memory。二者静态和persistent的理论驻留上限均2 CTA/SM，108 SM对应216 CTA的理论容量。108总CTA限制了可供调度的并行工作，未自动获得复用收益。理论容量不是实测occupancy。

唯一网格反事实在隔离DSO把上限108改216，保留同一tile/数学/归约/舍入；为同进程加载修正了模块名与C++符号隔离。三B14输入，每臂8轮旋转顺序、profiler之外CUDA event计时，中位数如下：

| KV | 静态 ms | P108 ms | P216 ms | P108比静态慢 | P216比静态慢 |
|---|---:|---:|---:|---:|---:|
| 288 | 2.603808 | 2.711968 | 2.622720 | 4.154% | 0.726% |
| 1152 | 2.479712 | 2.598272 | 2.491472 | 4.781% | 0.474% |
| 2160 | 2.881904 | 3.039216 | 2.887056 | 5.459% | 0.179% |

112次完整block调用，输出/全部有效KV逐位与RNG检查通过。未锁时钟，三case间绝对时间非单调，不能直接跨KV推断延迟趋势或把这次值替换M32两进程数据。只支持同case旋转对照的描述性判断；仍无新增加速。P216未替换生产optimized源码/DSO。

## HYPOTHESIS与支持程度

时间线加单变量网格干预支持：当前退化主要来自过小常驻grid限制并行度；将上限恢复到216大幅消除了退化。还保留每tile循环、排空/同步和FFN1寄存器增量；剩余0.18%–0.73%不能据此精确归因。没有跨阶段ready、公共数据新复用或专职warp分工，不能把此实验称为整层MegaTransformer收益。

整层绝对瓶颈与退化来源分开：原长B14 rep0中未缓存cross K/V投影约1427微秒、self KV pack约472微秒，未被这次常驻循环改变。后续还须对比已有强cached/shared-prefix/Graph路径，不能把普通缓存收益算作新调度贡献。

## UNKNOWN与失败

NCU第一个作业因ERR_NVGPUCTRPERM退出1，实际occupancy、DRAM、L2、Tensor Core/stall计数未取得；剩余五个NCU取消。内层4次block数值通过不等于NCU成功。不改系统权限/时钟，trace的估算occupancy不冒充硬件计数。

grid-counterfactual-1在14次调用后因Python同名扩展身份错误失败，P216数据无效；attempt2在第10次调用因CUDA invalid argument失败。独立PyInit和C++/CUDA命名空间后attempt3完整通过；符号/模板静态状态冲突是与修复一致的实施解释，未单独定位到具体CUDA API。原失败、计时和日志全部保留，不参与有效P216汇总。

## RECOMMENDATION

保留M32局部原型和静态对照。按用户要求转向单层统一kernel：补齐QKV/Wo/FFN2等设备原语，建立整层DAG、ready计数、warp组执行与workspace生命周期。现有host cuBLASLt不能直接在设备任务循环里调用；旧FFN2数值失败不降低门。先全层覆盖与状态正确，再优化流水/专职warp，不扩30层。

本轮五GPU作业生命周期81.856秒、293次block调用尝试（含失败）、0full30/0episode；三次CPU构建合计342.005秒。作业生命周期与CUDA active时间分开，后者UNKNOWN。登记至最终资源核查约28分钟，包括准备、构建、等待与传输。未用GPU预算不自动转授新任务。22:53:37北京时间两GPU均0MiB且无计算进程，自有PID+出生时间核验均未运行；新namespace约42.91MiB，含100份原始trace及失败，低于256MiB。

## 证据

- [逐次时间线摘要](TIMELINE-ROWS.csv)、[九点均值](TIMELINE-SUMMARY.json)、[反事实摘要](GRID-SUMMARY.json)。
- [原timeline结果](remote-evidence/timeline-1/results/RESULT.json)、[有效反事实结果](remote-evidence/grid-counterfactual-3/results/RESULT.json)。
- [NCU外层收据](remote-evidence/ncu-long-static-1/OWNED-RESULT.json)、[失败1](remote-evidence/grid-counterfactual-1/OWNED-RESULT.json)、[失败2](remote-evidence/grid-counterfactual-2/OWNED-RESULT.json)。
- [资源与止损](RESOURCE-AND-STOP.json)、[现场终态/trace索引](remote-evidence/FINAL-RESOURCE.json)、[源身份](SOURCE-FINAL.json)、[诊断副本补丁](remote-evidence/GRID216-REPRODUCE.patch)、[冻结核对](FROZEN-AFTER.json)。

原始trace在远端`/root/cxz/ops/single-block-profile-20261019`；未将tensor、DSO或工具链上传GitHub。四质量字段均false，生产TF/阈值/FAILED/默认A_off不变。
