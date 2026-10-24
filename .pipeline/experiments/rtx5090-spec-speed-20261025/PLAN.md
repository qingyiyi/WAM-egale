---
kind: experiment
cycle: wam-research-20261016
status: complete_with_limitations
---
# RTX5090上的投机侧速度与临时资产收口

本轮用户直接授权在现有5090测LIBERO/RoboTwin投机加速，并要求只回传数据，删除远端临时测试目录和权重。不是恢复过期队列或改变旧质量证书。Target、冻结TF与同权重PF2三臂；不将单层优化集成进去，不训练/调阈值。

- [x] P0：聚焦当前Cycle，读取实际资源、既有完整任务计时入口与口径，Maintain存储要求。现场5090空闲、数据盘可用45035552768B，GPU32GiB；现环境没有模拟器/完整server依赖。
- [x] P1：从现有BAAI资产按套迁入，固定源/权重/normalizer/bank/环境身份；不扫描模型SHA、不smoke。先Long及RoboTwin，再其余三LIBERO套件，单模型worker，按实际空间轮换。
- [x] P2（最终实际覆盖见RESULT）：沿源码完整调用测Target/TF/PF2，记录完整policy、action各round和有效scheduler更新g、实际路由/forward数；同卡、eager/no Graph、公平reset。完整请求重放只称推理E2E，不冒充机器人任务。
- [x] P3：若现有完整环境入口与依赖在本轮可运行，固定已暴露DEV面板做自然成功/原horizon的任务计时；不截断、不重跑已执行episode、不挑成功局。LIBERO优先每任务init49，RoboTwin每任务valid49，T/C/P Latin顺序。每套一轮仅为工程成本面板，不增加独立成功率N。
- [x] P4：独立分析成本、成功/失败与长度，汇报可测边界/未完成项；先回传核验，再逐套删权重，最后清理整测试目录与owned进程，HW与main镜像更新。

## 预算与范围

现有5090单卡，CPU最多8核（模型/环境各2–4，迁移/guard低优先级），最多一个模型worker/一个simulator。第一轮总生命周期上限10小时（含迁移、安装、等待、失败与清理），原A100160小时预算不动。本机仅编排/分析/回传，CPU≤2，无本机GPU；A100只读取资产，不额外使用A100 GPU或停止原质量队列。

单套准备上限90分钟、单模型就绪上限15分钟；每LIBERO完整局1200秒/4000teacher、RoboTwin1800秒/8000teacher；新完整自然episode至多138局（四LIBERO×10×3＋RoboTwin6×3）。完整请求重放有独立计数，至多每套三真实KV桶×三臂×3重复；没有足够完整fixture则报告缺项，不构造不存在条件。总target forward尝试上限600000，重放不是独立机器人局。

新增远端持久化峰值≤34GiB（首个teacher前前瞻修订，原30GiB登记保留）且始终留5GiB，单套权重临时驻留；必要环境、缓存也限本轮目录。本机回传原始结果/源码/必要证据≤3GiB；不得把模型复制回本机充当结果。每套结果正常回传和文件大小/rsync传输完整性核验后删远端该套资产；失败/timeout也是结果，收据照留。清理预留20分钟，不为达到某速度比延时。

## 证据与比较边界

默认四套LIBERO（Long、Goal、Spatial、Object）与RoboTwin当前六任务；先主Long/RoboTwin，不为填表移植新模型或重训。完全复用原冻结tau、树、selector、scheduler、PF2定义和normalizer；RoboTwin video25与LIBERO video20分别保留，action50。

当前新用户授权为独占硬件上的工程性能观察；旧engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission保持false，旧正式45chunk入口不调用。成功率与全部失败/长度并列，双方成功时间比仅补充。记录ΣT_round/Σg、完整policy时间/chunk、reset/cache、自然暖episode与冷装载；原始forward不按同成本相加。5090新数值路径可能改变接受，实际记录而非借A100计数或证书。

原有算子工作区/二进制与baseline不改；本轮目录`/root/autodl-tmp/cxz/spec-speed-20261025`独立，模型/HF/CUDA/临时下载不得写出此根。现有工具链只读复用，新依赖隔离到本轮venv。逐job登记目标/输入/开始/硬截止/存储/进程出生身份/终态/清理；结果回传失败不删唯一原件。

## 首次GPU前准备修订

匹配Python3.10模拟器依赖与RoboTwin网格资产所需空间，峰值上限由自定30GiB前瞻修订为34GiB，仍始终留5GiB；不购买/扩容。零teacher且零环境命令的准备/导入/资源绑定失败允许在同一未重置硬截止内，以新attempt修复接续，保留每次失败；执行过的完整任务尝试不自动重试。

## Long阶段结果

最终server-05/panel-06的26条已完整终态：task1仅T/P，其余task2..9三臂完整。共同8组三臂全部成功；动作成本加速C/P=1.2328/1.1465，完整policy/chunk=1.1444/1.0942，暖任务E2E=1.1333/1.0690，客户端生命周期=1.1207/1.0616。两条已执行PARTIAL不重跑，默认FSDP的旧task0/T不混入。只代表本小面板，四科学权限false。回传ACK和临时Long权重清理收据独立保存；RoboTwin随后同一有界链接续。
