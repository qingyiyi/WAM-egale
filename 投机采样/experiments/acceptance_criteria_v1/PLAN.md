# 固定 PF2 的三个接受规则比较

用户于2026-10-14同意继续上轮提出的比较，并要求结果、HW汇总与下一步。独立实验目录，不修改公共模型/原scheduler/旧实验/全局preset/protected HW。

- [x] 接受器出处、同状态一步偏差与旧q90来源已核验。
- [x] 复用冻结PF2：spec、spec、完整原target；bonus、A_off、target、video20/action50、执行频率不变。
- [ ] 三个CPU scorer、actual scheduler集成、源代码独立复核通过。
- [ ] 仅teacher train task0/1 init0/1校准通过数量；val8只报告，不调阈值。
- [ ] 冻结WL2/MSE/RES三种规则，20个完整GPU chunk数值/控制检查。
- [ ] T/PF2_WL2/PF2_MSE/PF2_RES，task0/1×init15/16，16个完整任务；不根据结果追加或改阈值。
- [ ] 独立从原始episode/初态/调用/树/周期日志重算、整理到HW与汇总。

WL2沿用旧阈值；MSE是FlowCast速度验证适配，仍用学习型双头draft，不称完整FlowCast复现。RES检查真实scheduler/projected candidate与同parent target更新的BF16状态差，使用DPM-Solver的max绝对/相对尺度(.0078,.05)并只校准统一倍率；不是原嵌入高阶求解器。额外3个scheduler probe不提交更新、无模型前向；计入诊断，不用共享GPU墙钟声称加速。

校准集只含已有OLD访问状态，不能据单步通过率声称新树或闭环保质。init15/16未用于当前teacher/draft选择，但曾出现在历史target-only；称冻结接受器后的配对初态pilot，不称全局未见test。成功/失败完整horizon800，无截断、失败按800步计。

预声明质量：对T及PF2_WL2分别检查成功丢失=0、成功数不降、总惩罚步数比≤1.05、最差配对变长比≤1.10；最大绝对变化另报。16例pilot不能证明总体成功率非劣。采用候选还需目标调用/推进有实际收益，global preset不自动更改。

新GPU硬限6300秒，累计36000秒；实际carry25684.616692936048秒/108894次target intents。新有限预约16×4000+20×80=65600；累计target cap174494。双模型双EGL、任务自有进程清理、显存容量/余量与来源门保留。允许其他GPU进程共用质量任务，性能测量另行隔离。

仿真瓶颈先用既有真实计时核查；不改变物理步、控制频率、图像采样或终止条件。固定新大输出在/var/tmp任务专用目录，项目内只留小汇总与位置指针。

CVPR2027全文11/16 AoE；老师稿内部硬节点北京时间11/01 18:00。MegaTransformer由用户自行设计、思路类似MegaMoE；本轮只排期，不把未实现kernel计为成果。
