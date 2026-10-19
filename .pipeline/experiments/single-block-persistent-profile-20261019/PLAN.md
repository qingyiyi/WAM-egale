# M33 单层persistent退化归因

用户已要求实际profile。只研究M32同一层现有实现；30层/机器人/生产配置不动。该原型尚未满足用户新澄清的整层统一persistent执行器目标。

- [x] 核资源、工具与固定源码，注册新预算；不借M32余额。
- [x] 真实九block×四模式：tiled、persistent，分别关闭/启用FFN1融合；torch profiler分离实际CUDA核与host/API间隙。每组合两次warmup、两次profile完整forward，输出逐位核对。
- [ ] 依据首轮trace，NCU采B14短/中/长，静态与persistent各一完整block的attention/FFN1核；显式clock-control none/cache-control none，profile时间不进正式性能表。
- [x] 检查寄存器/共享内存/理论和实际驻留、CTA/waves、L2/DRAM与stall；不以静态资源猜根因。必要时仅一个由证据支持的调度反事实，不无限搜索。
- [x] 给出整个单层统一persistent执行器的primitive覆盖、DAG、就绪/同步、workspace与精度缺口；不把阶段内CTA循环当成warp任务调度完成。
- [ ] 归档原始profile/失败/资源/清理并更新HW和main。

资源：远端GPU0，CPU4/5两线程；GPU生命周期900秒/512完整block/0full30/0episode，总墙钟60分钟，最后5分钟清理。原九个PT复用；新增持久证据≤256MiB。工具不支持或硬件计数权限不足则保留UNKNOWN，不修改系统设置。正式E2E仍本机后续。

## 22:34预登记：唯一网格反事实

NCU首个完整B14 long作业因ERR_NVGPUCTRPERM退出1，应用自身4次block精度通过不代表计数器采集成功；后五个NCU停止。旧失败保留，不改系统权限。

时间线显示静态/常驻同为34次kernel launch，耗时增量集中self/cross attention与FFN1；现常驻上限108个CTA。唯一挑战变体仅将上限改为2×SM=216，其余device函数、tile形状、归约/舍入和输入不动；不尝试第三个网格。只在M33隔离构建，不替换生产optimized源码/DSO。

三真实B14 KV，各比较static、P108、P216；每臂2warmup、2profile、8轮旋转顺序event计时，共112次完整block（含一次变体静态路径opt-in核对），叠加此前157=269≤512。每次核对输出、全部有效KV、RNG。runtime读取编译资源和理论active CTA上限，明确不是实测occupancy。单GPU job≤180秒，CPU构建≤240秒；沿用原900秒生命周期和23:20停工作，不追加预算。

## 22:50实施失败及同一反事实的修正

grid-counterfactual-1在14次完整block后缺少variant metadata接口，说明同名单阶段Python扩展未获得独立身份；该attempt所有P216观察无效，失败与成本保留。改为独立PyInit名称后，grid-counterfactual-2在第10次block调用遇到CUDA invalid argument；静态模板状态/host stub同符号的DSO隔离不足是待验证解释。只把诊断副本的C++/CUDA namespace改为独立名字，不改变数学或增加网格候选。

下一attempt仍static/P108/P216三臂同输入，112次完整block。累计上限153+4+14+10+112=293≤512，生命周期900秒、原23:20停止不变。若此身份隔离仍不能执行，停止反事实，报告UNKNOWN，不继续编译/重跑。生产源码/DSO不变。

## 终态

PROFILE_COMPLETE_COUNTERS_UNAVAILABLE；NCU未完成项因权限失败停止，实际counter UNKNOWN。[结果](RESULT.md)与全部失败/资源已归档。新M34整层实现尚未启动，M33没有剩余作业。
