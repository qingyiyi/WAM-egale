# 当前执行：FFN四组依赖流水，有限精度通过，整层收益很小

当前optimized采用R2：每组28个FFN1+GELU生产tile就绪后派24个FFN2半段任务；两K段可以并行MMA，第二段仅提交时等第一段。ACT仍落全局buffer，数学和BF16舍入不变。只有FFN局部流水，尚无整层通用DAG或固定warp角色流水。

|KV|同轮此前阶段版相对原Torch|当前R2相对原Torch|R2较此前阶段版延迟变化|
|---|---:|---:|---:|
|288|1.1607×|1.1646×|−0.342%|
|1152|1.6748×|1.6808×|−0.357%|
|2160|2.7295×|2.7386×|−0.332%|

比值=T_Torch/T_自有，两独立进程、同真实输入及最终Cross K/V预计算、远端A100-SXM4 warm完整单层Graph。两进程方向一致但幅度小，不外推30层或机器人E2E。9签名/36结构有限逐位及状态通过，四科学权限和production_admitted仍false。

debug确认组0/1/2在最后FFN1组完成前开始消费；最后生产和最后完成仍为组3，关键串行链保留。warm数学任务3400/3472/3400不变，grid29→27，新增发布/轮询成本未定量拆分。

R1过度串行化负结果保留。R2首验证因编译输出目录错误在0调用退出；R2产物正确安放，R1从原对象恢复且身份完全一致，再同截止新retry验证。FAILED、源版本、全部计时和原截止保留。

两机研究入口采用R2，本机未另测。18:10:40 owned全部退出，累计9844.678/36000秒；原本轮18:40:44截止不变。未恢复过期机器人队列，未新开GPU作业。

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)、[源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[阶段表](../算子研究/wam_ops/optimized/STAGE-MAP.md)、[HW](../.pipeline/PROGRESS.compact.md)。下一步先评估最后组关键路径与同步成本，再决定更细流水是否值得。
