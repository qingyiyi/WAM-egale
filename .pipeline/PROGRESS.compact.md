# 当前研究进度（2026-10-22T15:31:11.420822+08:00）

|ID|状态|证据与下一步|
|---|---|---|
|M39|in_progress|RMS+RoPE有限精度通过，完整层再快0.39%–0.44%，两机采用；仍慢最强guarded约9%–11%|
|M17|in_progress|LIBERO6000局齐，非独立TEST；暴露/首两S/长尾缺口|
|M40|in_progress|旧DEV90及工程4局归档；旧截止过期、扩大未批准|

## M39本轮

[RMS+RoPE融合](experiments/single-layer-async-gemm-20261020/rms-rope-fusion-20261022/RESULT.md)只改B14 self Q/K后处理，保留归约树、BF16舍入、FP64旋转、B2兼容与其余阶段同步。9签名81次、21结构84次完整层有限逐位/有效KV/RNG通过；普通/debug一致。

远端空闲A100，两进程五臂交错Graph，短/中/长完整层2.270464/2.434816/2.449152ms→2.261504/2.424576/2.438400ms，降0.395%/0.421%/0.439%。收益小，非显著性或E2E结论；仍慢同轮guarded9.32%/10.93%/10.86%。普通路径少2.625MiB中间张量、336tile、两grid同步，当前36次。未恢复ready或实现新epilogue流水。

源0f79579d、DSO795be814两机采用，旧调制DSO41af9a65与ready负结果保留。当前投影/attention QK/PV源码及活动二进制HMMA已核对，利用率未新测。资源108CTA/100096B shared/255regs/static local936B；静态local不等于动态流量。

CPU编译308.493秒；GPU worker94.133秒，含上传等待清理整窗153.757秒，累计7480.439/36000秒。15:28:28 owned全部退出，两卡GPU列表为空；新窗口15:55:54截止保留但作业已闭合，无自动接续。四GPU步骤无失败/timeout；早期只读收尾拒绝已记录。

[优化证据索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。下一步按用户讨论投影整行依赖或后处理流水；尚未实施GEMM+RMS/新调度，不自动启动新GPU。

## 其他主线与边界

LIBERO各方法4×10×50=2000，共6000局；T/TF/PF2成功数Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499（各500）。非三seed、非独立TEST，曝光/首两S/长尾缺口保留。

RoboTwin旧DEV90 T/C/P25/24/23（各30）；工程4局不新增独立质量N。累计54718.244/86400秒，10/22 01:26:04截止不变，1800局/160h未启动/未批准。

四科学权限false；baseline/投机/原tile/state/cycle/rules不动；旧FAILED/retention/paused72保留。BAAI仅~/cxz。10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包，证据不足收缩claim。

[本轮前状态](cycles/wam-research-20261016/history/before-modulation-result-20261022/PROGRESS.md)。
