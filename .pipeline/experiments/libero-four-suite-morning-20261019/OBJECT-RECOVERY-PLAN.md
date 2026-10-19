# Object续采输出冲突：修复与有限补采提案

观察：CURRENT_DATA的30局均有完整机器人终态；前16条collector谓词通过，新增row16–29全部collector_invalid/collector_errors失败。row16首chunk错误为immutable compact shard already exists，随后错误状态保持。COUNTER的调用/合法g等其他谓词通过。原PACK_CURRENT拒绝入训是有效保护，不能改PASS或拿旧latent标签补数据。

已准备可审阅补丁OBJECT-CAPTURE-ISOLATION-PROPOSAL.patch：仅新namespace的server入口将Collector写目录改为原compact根下全新job专用子目录。原保存数学、输入/标签、容量账本、严格断言全部保留；构造时已存在则失败，不覆盖；counter与collection审计路径仍为子目录真实路径。当前只做AST语法核验，未部署/未新启动GPU，不能写运行验证通过。

待授权范围：保留旧16有效条与全部旧14无效capture/终态，只对OBJECT-RECOVERY-CASES中的row16–29、同task/init/CONT3000/原FIT-CAL角色重采14完整episode；每局horizon800/最多4000 target attempts，无自动重试；上限56000新增forward、256MiB新增capture，准备/清理一并入账。可分两卡独立worker，绝不共享可变状态。14条是基础设施补采，不计新独立初态。原Object累计cap要显式加14局/56000尝试，不重置原M17起点/10月21日截止/总存储上限。

之后仅接续早已授权且尚未开始的ON3000与Object500组三臂，不重复其他500局、不启动TEST/120/架构搜索。先核现有预算余量/有效初态/所有数据准入，无效即停。原控制器fixed CPU step failed, no retry和迁移协议不允许自动补跑，所以本提案需要一次明确补采授权。没有该授权时只保留方案，远端两卡空闲事实不能写成正在训练。


## 明确授权后的执行

用户明确批准“允许补采14局并接续原任务”，已保存长期决定。实际实现采用object_capture_recovery_v3，原v1/v2/旧失败全部保留。v2首次启动在GPU创建/episode之前被过期T_POOL模板截止拦截；只修正NEW_MATCHED_TARGET200计划沿原M17总截止，旧训练T_POOL截止不改、不恢复。v3已在远端两张A100各启动独立完整worker，CURRENT14与既定MATCHED_TARGET200并行；数据准入后ON3000和全500由controller自动接续。

09:08实测两臂各2局完整，首补采counter_invalid=false、errors=0、10个shard完整，命令/模型正常运行；这是有限运行证据，不能代替全14数据审计。原新14无效消耗5876次已经扣除，新root剩余尝试上限6106124（6056000+56000−5876），原10/17起点及10/21 16:56:31停新、17:11:31清理保持。Controller PID553438出生697042563匹配，CPU只读health-v2约5分钟监督。

新增namespace源码/元数据小文件实际字节比较一致，无模型hash、smoke或单独新测试。旧16仅复制少量路径绑定元数据，tensor原件不变；新数据写原共享容量根下的新job子目录。Matched Target200不用于ON训练，前200完整评估使用其物理GPU；所有评测原512-token BF16 reset bank一致。
