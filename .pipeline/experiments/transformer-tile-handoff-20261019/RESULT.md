# M30：结构交接与MegaMoE思想设计

终态：COMPLETE_CPU_HANDOFF_NOT_GPU_PROTOTYPE。本轮GPU预算及实际使用均0，没有新训练、机器人episode、kernel/30层/chunk性能数字。源码基线固定99f7c2f3e845920fd83b22fedd19e72089f3a886，数学实现未修改。

## 已观察

- 对9份既有真实逻辑fixture读取CPU元数据：B2/B14 action、B2 video各短中长；只解码小metadata，不加载完整target到GPU、不新保存tensor。审查程序最终运行约0.697秒CPU墙钟，不等于整个M30生命周期。
- 从safetensors header读取block0的27参数形状/dtype、原配置D3072/H24/FFN14336/30层；没有读取权重payload或做模型hash。
- action有效K为288/1152/2160；本次video对应272/1136/2144。KV物理pool容量2160不等于每次可见长度。B14 attention仍各Q16，只有同CFG历史共享，private KV不共享。
- 机器可读GEMM账本9行，源码推导DAG21节点；7个小源码身份保存，不把逻辑DAG称为runtime kernel trace。
- 固定官方MegaMoE/DeepGEMM commit，下载10份源码/许可证只读，未安装/执行。实际有BF16实现，但该路径为SM100，不是现成A100实现。

## 设计建议与未知

建议先做显式tile坐标的FFN1+GELU常驻worker；FFN2严格设备原语门通过后，接入行块ready的FFN1/GELU→FFN2→残差子图，再逐项研究K块流水和warp专门化。具体tile/策略与增量仍为假设；当前M28动态路径也未与最强B14/cached/Graph统一。

缺口：生产CUDA原stride/storage offset/alias未由contiguous CPU fixture恢复；FFN2旧device逐位失败未解决；新任务队列、persistent kernel、candidate-aware增量、ΣT_round/Σg和闭环收益均未测。保留所有旧FAILED与四项false。

## CPU尝试及资源

首个metadata提取因压缩字段是dict而非Tensor，调用sum失败；修正为仅解码≤10000元素的metadata，第二次退出0。该错误属于CPU审查工具，不能改称机器人算法失败或删旧任务。源派生脚本随后退出0，未运行GPU或外部源码。

首次保护文件核对错误地拿省略了state/rules的公开镜像作参考，assert失败；该尝试保留。随后改用M28本机已保存的源身份：1875投机源、92算子数学文件和保护文件均未变，不创建缺失cycle.yaml。未把“公开未收录”解释为本机文件被修改。

生命周期/存储/边界核对见[RESOURCE-AND-STOP.json](RESOURCE-AND-STOP.json)。本轮无持有GPU上下文的owned worker，不向既有远端队列或他人进程发送信号。远端质量继续原M17授权，不计作M30实验产出。

## 入口

- [可直接转交协作者的结构/shape/源码](HANDOFF.md)
- [综合HW的具体设计](DESIGN.md)
- [全部工作依赖、日程与止损](WORK-OUTLINE.md)
- [MegaMoE源机制与SM80映射](MEGAMOE-NOTES.md)、[源清单](MEGAMOE-REFERENCE.json)
- [SHAPES](SHAPES.json)、[PARAMETERS](PARAMETERS.json)、[GEMM-SHAPES](GEMM-SHAPES.json)、[BLOCK-DAG](BLOCK-DAG.json)
- [已完成三套件成功率](../libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)：另属既有质量线的冻结摘要，不是本轮新运行。
