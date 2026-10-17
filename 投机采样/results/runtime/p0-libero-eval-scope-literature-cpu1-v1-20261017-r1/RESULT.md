# LIBERO90是否必要、Long10是否足够

结论：本轮不宜把LIBERO90作为论文必做依赖。优先完整收口Long10；仅Long10能否充分支撑最终CVPR主张，取决于质量、强baseline、因果消融和系统收益，不作录用保证。暂不将用户问句解释为已经取消90授权，更不启动无可信Target的90训练。

补充核验：LIBERO官方README明确将LIBERO-100拆为用于policy预训练的90和用于下游终身学习评测的10。这个原始设计不禁止评测90，但直接反驳“必须评测100种任务才符合通常LIBERO论文协议”的假设。现代VLA常见的Spatial/Object/Goal/Long四套各10任务，也不等于LIBERO-100。官方来源：https://github.com/Lifelong-Robot-Learning/LIBERO#benchmarking-knowledge-transfer-for-lifelong-robot-learning 。

| 一手资料 | 实际范围 | 对本项目的含义 |
|---|---|---|
| LingBot-VA原文2601.21998v1 §4.3.2/Table2 | Spatial/Object/Goal/Long各10task；每套每seed500评测，三seed | 90不是它的主评测，但原文也不是只做Long |
| CoT-VLA，CVPR2025；作者项目页LIBERO表说明 | 同四套；500初态×三轮 | 常见论文协议是四套小suite，非90；当前Long500/arm仅固定一个eval seed |
| Seer，作者官方repo标ICLR2025 Oral | LIBERO部分Long，另外CALVIN和真机 | Long可作为LIBERO主实验，不能把多环境论文误读成仅Long即可 |
| Realtime-VLA FLASH，2605.13778v1 §4.1/4.2 | 四套各10task×每task50；每套另训draft；另真机 | 最接近投机方向不跑90；预印本，未拿它冒充CCF-A准入规范 |

建议默认主线：当前冻结LingBot-VA Long checkpoint+冻结TF_ON的Target/C/P全Long500配对；保留全部失败、800罚分和实际长度，报告双方成功分布和全部成本；单位有效提交ΣT_round/Σg、完整chunk、任务与冷/暖E2E分层。主线质量及新增调度证据若不成立，扩90不能修复。若主线成立并有预算，再考虑一个具名社区Target的小套件作为外推检查，需另登记，不自动把90改成三个套件全跑。

按50初态×三臂，Long1500局，90另13500局，省去90可减少Long+90计划90%的episode数；任务长短、失败长尾不同，不保证同比墙钟减少。还减少Target获取/适配、90提示bank、teacher采集、独立TF训练和验收工作。可靠90target的独立检索仍未找到，详见 ../p0-libero90-checkpoint-reliability-deepsearch-cpu1-v1-20261017-r1/RELIABILITY-RESULT.md。

主张边界：只在一个Long目标模型验证，不能写跨WAM/跨环境普适性；Long全10task不是10条episode。当前已冻结生产候选再做全量描述，不将已用于FIT/DEV或原预留40–49合并称独立TEST。四权限、旧NO_GO/FAILED不变。

原文和实际读取材料：
- https://arxiv.org/html/2601.21998v1 （完整HTML已读，缓存lingbot-va.html/.txt）
- https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html （正式会议信息已读）
- https://cot-vla.github.io/ （作者主表说明已读，cot-vla-project.html）
- https://github.com/InternRobotics/Seer （官方代码/实验导航已读，seer-readme.md；仓库API简介ICLR2025 Oral）
- https://arxiv.org/html/2605.13778v1 （完整HTML已读，flash.html/.txt）

工具失败保留：web检索服务连接两次失败，改直接读作者/会议官网。CoT-VLA PDF获取截断/转换失败，不据该残件声称读完整PDF；协议来自成功读取的作者项目页。初始bs4缺失，未安装依赖，改stdlib解析完整HTML。所有工作CPU一核/零GPU，无模型、测试或SHA。文献来源是评测范围证据，不是安全阈值/质量保证。

2026-10-17 15:20后浏览服务恢复，重新打开LIBERO官方、CoT-VLA作者页、LingBot-VA/FLASH固定v1和Seer官方源码。LingBot-VA的定位更正为§4.3.2，不沿用早先笔记误写的§4.1；评测范围和计数结论未变。后续成功读取不覆盖前述失败记录。
