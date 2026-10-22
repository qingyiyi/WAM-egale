# WAM短状态（2026-10-22T14:42:31.869291+08:00）

|ID|状态|证据与下一步|
|---|---|---|
|M39|in_progress|按需调制融合通过有限精度、整层快约1%；当前阶段同步版，下一项先讨论Norm1与输入调制|
|M17|in_progress|LIBERO6000局齐，非独立TEST；暴露/首两S/长尾缺口|
|M40|in_progress|工程4局与旧DEV90已归档；旧截止过期，扩大未批准|

## M39当前结果

[按需调制结果](experiments/single-layer-async-gemm-20261020/modulation-ondemand-20261022/RESULT.md)：9签名81次、21结构84次完整层有限逐位/有效KV/输入/RNG通过；普通/debug一致。普通路径取消15.75MiB MOD张量，tile13928/14000→11912/11984，grid同步39→38。未恢复ready队列。

远端独占A100、两独立进程五臂交错Graph，短/中/长旧11 2.292/2.459/2.472ms→新版2.269/2.435/2.449ms，下降1.005%/0.979%/0.942%。相对同轮最强guarded仍慢9.59%/11.34%/11.27%，不是完全对齐或机器人E2E。两机optimized当前source9edf85a2、DSO41af9a65；旧11与ready负结果保留。

本轮CPU编译303.392秒，GPU worker94.626秒；含上传等待整窗保守计费430.874秒，累计7326.682/36000秒，CUDA active未测。14:38:38 owned全部退出、两卡GPU列表为空；新截止15:01:27保留但本轮已闭合，不自动新跑。传输公钥失败及两55秒超时保留，GPU作业无失败。

下一项按用户逐条讨论Norm1/输入调制合并是否值得做，尚未实施。ready上一轮慢56%–64%已回退，不能把新约1%收益归于ready调度。

## 其他主线与边界

LIBERO各方法4×10×50=2000，共6000局；T/TF/PF2成功数Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499（各500）。非三seed、非独立TEST，曝光/首两S/长尾缺口保留。

RoboTwin旧DEV90 T/C/P25/24/23（各30）；工程4局不新增独立质量N。累计54718.244/86400秒，10/22 01:26:04截止不变，1800局/160h未启动/未批准。

四科学权限false；baseline/投机/原tile/state/cycle/rules不动；旧FAILED/retention/paused72保留。BAAI仅~/cxz。10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包，证据不足收缩claim。

[本轮前状态](cycles/wam-research-20261016/history/before-modulation-result-20261022/PROGRESS.md)。
