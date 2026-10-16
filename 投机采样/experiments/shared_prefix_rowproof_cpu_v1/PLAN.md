# 短KV：固定候选行映射证明复用的CPU原型

用户授权继续上一轮CPU优化方案并完成后汇报。仅新增本私有实验目录；原Torch参考、共享CUDA v2、历史KV访问和已有失败/计时全部保留。没有后台GPU任务，CPU检查由当前主代理执行并跟踪。

## 单一改动与边界

原共享adapter每个lease的30层都对相同候选行映射执行一次`bool(((rows>=0)&(rows<2)).all())`。原型保留第一次实际值域检查；后续层只在相同lease/request/generation/epoch、tensor身份/地址/版本/shape/stride/device/dtype、固定CFG2及原canonical row_values条件下复用该证明。各层shape/device/dtype继续检查。

原投影前后两次`_precheck`、五个历史张量key、slot创建/版本检查、模型状态/RNG、原数学计算与kernel全部继承，未移动或减少。活跃行映射发生跟踪到的Torch写入、缓存/模型失效或lease结束时，证明不能继续使用。与既有版本计数边界相同，不声称能检测绕过Torch版本计数的任意底层存储写入。

新`row_proof_enabled`默认false；旧环境变量不启用。`enabled`与`allow_cuda`仍各自默认false。R和HC始终走原行检查，S0表示原共享路径，S1表示开启新证明复用。证明是固定大小的主机元数据，没有新history14/fullKV14或GPU缓存；没有修改文本prompt投影/cross-attention。

## CPU阶段交付

使用实际公开生产loader创建可销毁的30层随机小CPU模型，比较S0/S1完整变化输入的输出字节，原R接口保持原数值检查。CPU profiler只统计行检查scope内的操作，预计取值范围检查及scalar从30次降为1次；不把小CPU时间或操作数量当作A100加速比。

检查default-off、R/HC分支、三档长度、候选/CFG隔离、行写入/替换、Q投影hook期间突变、五类历史张量写入、私有slots写入、显式恢复/回滚、模型/训练/backend变化、lease异常/释放/新generation、线程和CUDA边界。所有GPU模型调用/直接attention/EGL/episode/训练仍为0。

## 下一轮GPU验证设计（未执行、未授权、未排队）

建议采用同一真实短档KV288、完整30层/27个自然query，比较R原Torch、S0原共享、S1证明复用。保留全部query预热，六轮使用R/S0/S1全部六种顺序，各臂每个位置恰好出现两次；这样能直接判断改动是否优于同轮S0，不能只与上一轮的计时比较。

预算设计：capture50＋resource probe1＋三臂预热81＋六轮三臂计时486＋条件诊断最多3＋三臂收尾3＝最多624次target；1worker/1GPU启动/无重试，拟定含加载/IO/清理600秒。逐位原Torch、状态/RNG、lease/资源门保留；三组比较R/S0、R/S1、S0/S1均保留六组和≤7%MAD，不丢弃负样本。非稳定正向时诊断只解释机制，不替代OFF。

当前累计1865/2200次、2007.1380846686661/2700秒，剩335次/692.8619153313339秒。该完整三臂设计最多累计2489次/2607.138084668666秒，需在可执行GPU包装及CPU预检完成后另确认累计target上限至少2489（建议2500）；累计GPU仍可保持45分钟。现有回执只覆盖此前补测，不自动覆盖此新实验。本CPU原型不提供GPU启动入口，不能宣称GPU验证已经准备就绪。

CPU阶段结束后汇报机制与检查结果；短档稳定收益、原始GPU逐位结果、文本prompt共享及任务成功率/E2E仍待后续验证。不得用CPU随机模型结果替代训练目标GPU门，也不得从一档fixture推断所有短长度。
