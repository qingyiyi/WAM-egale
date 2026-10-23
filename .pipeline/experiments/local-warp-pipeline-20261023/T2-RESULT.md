# T2 FFN2局部准备结果

OBSERVED：B/C同一正常无插桩DSO e88283e143e17d02f5d9f5f431e3a61603f27c624dcfa8641e1bd8fbc03c94f1；CUDA源b17717d8c2331974798d17d965fcdfaca5ab11e1a0d2f7d2d0f03c1267d34abf。A独立原二进制。正常两臂资源均[108,1,112384,255,1016,256]，7KiB prep未增加原union峰值。B串行准备+C重叠准备；原4warp MMA/8warp提交、tile/K分段/partial发布不变。

每臂9真实81调用＋36结构144调用，有限逐位/有效KV/输入/RNG/普通debug/B2合法状态通过。另42诊断调用三KV核对独立A诊断副本与A全部有效中间量、B/C第一段partial/最终F2/output逐位相同。初诊断将logical_slots未初始化尾部一起比较而失败；新attempt明确保存raw仅logical_slots差异，并核对完整有效slot，未更改数值容差。

三KV正常C/A=1.000821/0.999460/1.001044；C/B=0.999063/0.997600/0.999128。C对A变化仅+0.082%/−0.054%/+0.104%，小于/接近本轮A/A观察波动；没有可信净收益。逐点INCONCLUSIVE，不采用C，保留A。不扩参数搜索。

软件诊断每KV6次、每次48个第二段任务：B代表warp准备/MMA区间重叠0/288；C重叠288/288，重叠P50约3.072–4.096us。只证明这些代表warp的区间重叠，不等于完整CTA关键路径缩短或TensorCore利用率上升。诊断BC静态local1048B，正常1016B；其MMA区间约171us不能冒充正常版时间或与旧92us直接比较。

控制限制：B的串行会合也出现在没有有效准备的第一段；C/B改善包含这次barrier的去除，不能完全归因提前准备。两者相同二进制/寄存器和shared不消除这项指令差异。故仅把重叠作为机制观察，不宣称干净的加速因果分解。最终采用依据独立A的无插桩C/A，当前不成立。

资源及raw见RESOURCE-LEDGER、T2-timing1/2、T2-DIAGNOSTIC-SUMMARY与原始DIAGNOSTIC。下一项独立于A做LN1调制融合，不组合未采用C。
