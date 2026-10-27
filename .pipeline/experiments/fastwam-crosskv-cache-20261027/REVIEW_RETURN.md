# Cross KV缓存：实现完成，保留研究入口

决定：接受chunk范围的显式缓存接口用于算子研究，严格eager数学保持；投机生产入口和全局默认不切换。官方compile缓存候选有额外数值变化，尚未通过闭环质量验证。

## OBSERVED

原PromptCache只缓存T5文本输出，不是Cross KV。当前proprio每chunk追加，因此本轮缓存生命周期严格限于一次policy请求；B1/B7分别建立。每个shape只计算一次action text_embedding和30层normalized K/V；之后各去噪调用直接读。自有persistent跳过原12/14阶段及其grid barrier，其余TMA/1T4M3E/mbarrier/严格BF16顺序不改。

精度：layer0/15/29 × B1/B7六组全部有效中间量、缓存、输入与CUDA RNG逐位通过；四固定Long旧DEV观测×Target/真实投机共八项完整30层/10步调用，缓存Torch及自有动作/接受分支/分数逐位一致，CPU Torch/CUDA RNG与条件KV不变。12个不同实际B7树查询各与两缓存引擎比较，24条teacher比较全部exact（原STATUS字段actual_B7_queries=24表示比较条数，不能当24棵独立树）。四观测来自一个task、两个init，不是四个独立episode。

B1与B7的原Torch K/V已测得非逐位相同，不能仅广播B1缓存；B7行间相同也不授权改变batch的GEMM。缓存占用B1=47,818,752B（45.6MiB），B1+B7=382,550,016B（364.8MiB）；native另需本轮显式视觉KV contiguous materialization 252,887,040B（241.2MiB），其时间只计入native，不转嫁Torch。

|完整policy平均ms|Target|真实投机|
|---|---:|---:|
|原action eager|388.148|247.642|
|同缓存action eager|350.610|235.336|
|自有persistent＋缓存|184.124|177.400|
|官方action compile|162.990|151.074|
|缓存action compile|153.649|148.778|

四输入各10次去首留9、交错引擎顺序、5090单作业；均使用共同prompt缓存及官方compiled视觉prefill，计入图像预处理、VAE、视觉prefill、condition/CrossKV构建、十步scheduler及动作后处理。初次编译与prompt填充单列；不含模型加载和机器人环境，不称任务E2E或success throughput。这里“eager”只指action，初始数值审查含视觉也全eager。

缓存对同引擎完整chunk：eager Target 1.107×、投机 1.052×；compile Target 1.061×、投机 1.015×。投机compiled的1.5%只是本轮观察，未跨进程独立复现，不能称稳健大收益。

双方应用缓存后，自有完整policy相对cached eager为Target 1.904×、投机 1.327×；相对cached compile却为0.834×/0.839×，即慢约19–20%。不能声称胜过官方强baseline。

warm完整层B1/B7：native cached 207.733/463.804µs，未缓存261.824/602.638µs，分别1.260/1.299×；相对同cache eager 1.407/.925×，相对同cache compile .605/.459×。设备计时使用两方对称Graph隔离host，直接API无外部Graph另见WHOLE.json；层编译入口关内部Graph，不冒充官方reduce-overhead部署。逐stage隔离时间不可相加为全层，完整层未插桩。cache构建82.610/113.860µs每层（不含text_embedding）另列。

compile缓存与原compile完整动作maxabs=.0078125、relL2=.2499–.3717%；四个实际投机观测路径相同，不等于所有接受边分数不变，也不等于新质量通过。纯eager缓存逐位精度不授权compile改变计算图后自动继承。

## HYPOTHESIS / UNKNOWN

从调用计数看：Target每chunk十个B1调用，投机三次B7加2–3次B1，缓存可复用次数较少且首次须建两套；VAE和视觉prefill未被该优化减少。这与投机chunk收益较小一致，不替代完整profile归因。共同优化也缩小本四输入的compile投机/Target加速：原约1.079×，缓存后约1.033×。不是全量任务速度。

UNKNOWN：跨任务/更多初态泛化、完整闭环成功率与自然trace、compile新版接受边翻转及独立进程稳定性；本轮没有环境实验，四科学权限均false。

## RECOMMENDATION

保留显式chunk缓存入口，后续算子和Torch使用同等缓存。下一步优先核查自有B7视觉KV布局/重复复制与32个输出tile的投影、FFN2并行度，不再靠取消BF16边界换局部数字。cache/compile版本先单独冻结身份和数值路径，生产采用需与投机会话协调质量验证。

## 固定证据入口

- SOURCE-MANIFEST.json、source-final/optimized、binary/fastwam_sm120_crosskv_cuda.so：本轮源码与DSO，本地小文件指纹；原严格独立DSO在前轮fastwam-tma-warp-port-20261027。
- layer-attempt01/results/{PRECISION,STAGES,WHOLE,STATUS}.json：层/全部有效buffer、阶段和全层同轮原始计时。
- policy-attempt01/results/{PRECISION,TEACHER-QUERIES,POLICY-TIMINGS,INPUT-MANIFEST,STATUS}.json：真实训练draft、接受路径、完整policy。原任务身份证明在输入manifest；未重命名为fresh TEST。
- STAGE-TABLE.md、SOURCE-AUDIT.md、RESOURCE-AND-STOP-REPORT.md。
