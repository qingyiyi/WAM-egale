# Cross K/V分组复用：有限精度通过，仍需匹配预计算基线

更新2026-10-22T16:12:24.664127+08:00。本轮在optimized中显式建立context映射，B14中相同条件的七个候选共用一组Cross K/V，每层保留正负CFG两组。self-attention历史映射独立，私有KV不共享；RMS依旧是独立阶段，未融合GEMM tile与RMS。两机研究入口采用新DSO，生产、baseline与投机代码不变。

## OBSERVED：完整层计时

|历史KV|原自有 ms|新分组复用 ms|延迟下降|guarded同样分组 ms|guarded已缓存 ms|
|---|---:|---:|---:|---:|---:|
|288|2.258176|1.205248|46.63%|1.225984|0.958464|
|1152|2.424064|1.372416|43.38%|1.344512|1.077760|
|2160|2.442496|1.388032|43.17%|1.359616|1.091072|


远端A100-SXM4-40GB GPU0，debug=false完整单层CUDA Graph；两独立进程、每臂预热600次、24轮交错，表为两进程中位数均值。七臂，分别为原Torch静态布局、guarded每次重算、guarded缓存投影、旧自有、同新DSO但不复用、新DSO分组复用、guarded分组重算。全部原始轮次和进程均保留，未挑最好轮。[CSV](WHOLE-LAYER-TIMING.csv)、[汇总](GPU-SUMMARY.json)、[进程1](TIMING-1.json)、[进程2](TIMING-2.json)。准入及case边界无外部GPU进程，计时前后SM1410MHz；没有修改时钟。

复用相对旧自有降低约43%–47%延迟，但大部分收益来自消除重复计算，不能算作新的tile调度收益。与同样每次仅计算两组条件的guarded对照相比，短KV快约1.69%，中/长慢约2.08%/2.09%。grouped Torch适配器仍需展开K/V给原attention，这部分计入；它不是最优分组attention的证明。

已有guarded缓存原形状K/V投影后更快：新自有仍慢约25.7%–27.3%。该缓存臂尚未缓存K RMS，冷填充在计时外；这行显示已有强基线边界，不是对等缓存条件下的新调度比较。当前自有每次仍重算两组K/V及RMS，未实现跨调用预计算。因此不能宣称超过最强B14缓存路径。

同新DSO关闭复用约2.270/2.437/2.450ms，比旧DSO慢0.52%/0.54%/0.29%。保留此对照，不能将所有二进制改变都归为零开销。group admission与静态layout准备在Graph外，未测动态Python/冷启动/完整30层/机器人E2E。

## OBSERVED：有限精度与状态

9真实签名81次完整层调用：B2 action、B14 action、B2 video各短/中/长。输出与23共有中间量逐位一致，max_abs=0；有效KV、输入/RNG、普通/debug、重复、B2合法cache转移通过。

另27结构用例108次：三KV各原布局、有效长度尾块/空洞、历史CFG换序、文本条件交错/换序。context映射始终与history映射分开，有限核验通过。原Torch的分组适配器三KV6次完整层输出/有效cache也通过，所以纳入计时。计时现场再核对全部臂输出/输入/有效cache/RNG。

[9签名](SIMPLE-GATE-RESULT.json)、[27结构](STRUCTURAL-RESULT.json)、[分组baseline](BASELINE-GATE.json)。这是有限fixture验证，不证明所有在线输入，也不继承机器人质量证书；四科学权限仍false。

## 实现与阶段变化

`whole_layer.py:prepare_context`在Graph外按context原始字节发现重复组，保存context_rows和context_roots。内容变化必须重新调用；不基于self历史映射，也不是自动生产cache协议。未prepare或无重复时保持完整形状路径。

`csrc/whole_layer.cu:context_linear`从每组代表行读取context，K/V写入紧凑组buffer。K RMS仍按原完整3072维归约及舍入执行。`csrc/fa2_stage.cuh:attention_dual`让每个候选Q读取所属组K/V，仍是14个独立Q16，没有跨候选注意。normal不展开K/V，debug为对照展开。

Cross K/V各1344→192逻辑tile，Cross K RMS3584→512；合计G2 11576→6200，G1 11648→6272。仍25数学阶段、36个全grid同步。三份Cross中间buffer各42→6MiB，总分配126→18MiB；这是张量分配推导，不是实测峰值或HBM流量。108CTA、256线程、100096B shared、255寄存器、static local936B；资源与前版相同，未新采动态spill/occupancy/profile。

[当前阶段表](CANDIDATE-STAGE-MAP.md)、[源码差异](CHANGE.patch)、[源身份](SOURCE-MANIFEST.json)。whole_layer.cu `bb5b63ed02a4c726abd960f5bb8d616d59410c342ec866e67481b7b27dd07357`；DSO `cc2656e76444a7b9f32ec29acdbf57ba94d9c5d0120f1961eb9f7a5d130e1593`。旧DSO `795be814c40a152583e13a34f6cf6bb5510b1fc38f5b4cd8c26076b4dc4efd75`及全部旧结果保留。

## RECOMMENDATION与UNKNOWN

优先在条件建立时预计算两组最终Cross K/V，连同K RMS缓存，每层独立；给原Torch/guarded同等能力，分别测冷构建与warm完整层，再分析剩余投影/attention/FFN调度。本轮未实现跨调用缓存，未把用户问句登记成新执行决定。

[LingBot-VA2基线核验](LINGBOT-VA2-ACCELERATION-BASELINE.md)：其约6.5倍以BF16 PyTorch为起点，累积蒸馏、FP8 TensorRT、FlashInfer和运行时优化；原文明确有Cross K/V预计算，未单列该项收益。不能据其数字外推本项目。

[资源与清理](RESOURCE-AND-STOP-REPORT.md)：CPU编译298.519秒，GPU worker111.688秒，整窗218.812秒，累计7699.251/36000秒，已清理。当前没有新机器人作业或生产集成。
