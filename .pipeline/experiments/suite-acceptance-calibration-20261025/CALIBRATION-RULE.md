# 读取误差分布前冻结的有限候选规则

主CAL为各套现有CURRENT/observer CAL，生成策略明确是CONT3000，不冒充所评ON3000的on-policy数据。它必须在episode/初态身份上与训练FIT全链不相交，训练使用固定最终checkpoint且没有CAL选权重/seed。已有完整扩大DEV暴露如实标注，不能称新独立TEST。

所有套件使用相同规则：当前冻结ON3000的生产predict接口，逐query按实际输入dtype执行，模型FP32、输出保持生产输入dtype，打分FP32；两个方向头取较小的abs(delta_sigma)*L2。每task等权、task内episode等权、episode内query等权，用逆经验CDF的q75和q90给出两个新tau，加上原tau，总计最多三个值。重复tau去重，不在看DEV后追加阈值。q90仅定义经验分位数，不追求90%在线树接受率或承诺90%任务成功。

Target CAL作为分布审计的次要面板，不参与本轮主tau选择；尤其Long旧Target准确历史权重字节身份缺失。缺少输入dtype字段的池必须沿原部署校准来源核查，无法确认则不用于生产候选；不得默认为CPU/GPU等价。

固定权重、normalizer、原步长缩放L2、K2/B2/bonus/prefix_then_cost、O1/O2/O3和每套video/action schedule。主DEV固定PFoff，只变化tau；旧PF2数据不混作新tau对照，双A100原PF2队列不改。

完整DEV使用每套LIBERO十任务的init49、Robo固定六任务valid49，均为已暴露DEV；完整运行至成功或原horizon。每case新运行Target、旧tau、q75、q90，逐case轮换顺序，同卡同软件、相同初始化/RNG与原动作后处理；不以旧5090或A100结果代替本机新配对。最多184完整局；源码导入/资源失效单列，不重复有teacher执行的失败attempt。

候选留存顺序：先成功数不低于配对Target及旧tau、无新增Target成功/候选失败；再检查原失败罚分和双方成功长度P50<=1.10、P90<=1.20、max<=1.30；符合时比较包含失败成本的policy/任务成本，等同优先较保守且改动小者。匹配MLP及独立TEST未包含，不据此自动放行旧完整质量门或正式性能。没有候选通过就保留旧默认并报告NO_GO，预算不足报告PARTIAL。
