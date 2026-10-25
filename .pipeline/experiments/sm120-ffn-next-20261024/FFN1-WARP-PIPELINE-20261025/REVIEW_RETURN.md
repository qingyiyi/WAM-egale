# FFN1 mbarrier流水：精度通过，保留S

## OBSERVED

实现并测量W：FFN1使用W0 TMA、W4–7 MMA、W1–3 epilogue；M64N128输出tile×448，每tile48个K64，完整K3072后bias、BF16舍入、GELU、BF16 ACT写global。四warp的MMA保持递增K16的数值顺序。输入双槽与FP32结果单槽独立，不alias。输入由TMA完成事务发布READY，四MMA warp读入寄存器后发布EMPTY；完整输出tile发布result_full，三个E warp消费后发布result_empty。热循环无全CTA同步，init/drain及FFN阶段边界仍同步。

TMA只负责global到shared输入，输出目前由epilogue线程写global；没有实现TMA store。每CTA静态承接2或3块，允许下一块MMA与上一块GELU/写回交叠；源码允许重叠不等于已测Tensor Core/CUDA Core同周期重叠比例。

当前S独立二进制保留。FFN2 partition64及commit64函数逐字节相同，七段BF16递推不改；ACT覆盖计数根据新真实tile变为四组112，FFN2只是期望完成计数随之更新。没有伪装原28/28/28/36计数通过。

两独立进程三KV的27捕获tensor、七段FP32 partial/BF16 running、有效KV、输入、CPU/CUDA RNG、非默认stream、计时前后、profile输出全部有限逐位通过；各CTA的实际1T/4M/3E循环计数通过。两进程完整层全部比S慢0.73%–0.88%，因此不采用W，恢复S；没有修改阈值、质量门、投机代码或baseline。

W FFN1对原Torch约0.781–0.787倍，FFN2约1.022–1.035倍；完整层与18阶段见STAGE-TABLE.md和SUMMARY.json。正常资源仍170CTA/1CTA每SM/100352B shared/255regs每线程/784B静态local，不能用occupancy下降解释退化。

## 源码支持的代价与UNKNOWN

- 为容纳4个MMA warp及独立缓冲，tile从108个128×256主体＋12个32×256尾块变为448个64×128。逻辑A/B搬运按source tile累计分别从83.25/180MiB变为168/336MiB；这是软件请求覆盖量，不是HBM实测，缓存可复用，不能宣称HBM翻倍。
- 新版每CTA只承接2–3块，170个CTA最终tile的epilogue必须排空；448块中最多278块的epilogue具备与同CTA下一块MMA重叠的机会。此为调度结构上限，不是实际隐藏比例。
- 新版还增加generic-to-async发布后的入口grid屏障、每K输入握手和多tile结束覆盖发布。GELU处理由256线程变96线程；这些变化与tile/并行度同时改变，不能单凭同场净差归因某一项。
- 未取得硬件计数器或各角色实际活动/等待时长，TMA、shared load、MMA、GELU和握手的各自份额仍UNKNOWN。旧S阶段值来自前一job，不能把跨轮阶段差当严格单因素比较。

## RECOMMENDATION

保留S；W作为正确可运行的局部mbarrier流水证据。下一次若继续，先在相同tile下分离输入/结果等待与GELU写回成本，再决定是否值得迁移TMA store。用户提出的E写BF16 shared输出槽、TMA负责global写回属于讨论中的新候选；需要单独输出槽、proxy同步及异步store完成/复用协议，不是把原store换成一条TMA指令。本轮未实施，不追加第二个设计。

四项科学权限保持false，有限block replay不代表闭环验证。

## 固定入口

- 当前保留S：`/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`。
- 本轮W冻结源码：本目录`whole_layer.cu`；S快照`S-whole_layer.cu`及`S.so`。
- `build/normal`与`build/profile`是独立W二进制；`make_profile.py`生成观察器，不覆盖正常源码。
- `measure-1/RESULT.json`、`measure-2/RESULT.json`保存每次计时、精度和计数；`measure-1/numeric-candidate-*.pt`保存七段数值。
- `JOB.json`、`RESOURCE-AND-STOP-REPORT.json`、`OWNED-CLEANUP.json`记录预算、终态及真实退出；不补造旧清理。

收尾：实际public入口S三fixture正常/debug/状态复验通过；2865调用、14.79分钟生命周期、约137.47MiB本轮产物，六phase均真实退出，无owned GPU进程；9项关键文件直接字节比对一致。
