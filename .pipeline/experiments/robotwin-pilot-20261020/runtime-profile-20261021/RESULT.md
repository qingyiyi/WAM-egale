# RoboTwin为什么慢：90局账本与一条完整成功任务profile

## 结论

**模型reset已经很小；当前主要成本是模型推理、逐动作双臂规划、光追观察，以及失败长尾。环境初始化仍有约19秒/局可研究。先核验工程改进，再按新吞吐申请扩大预算；1800局和160 GPU小时尚未启动/批准。**

## 90局真实成本（未施加本轮cProfile）

客户端生命周期合计23653.047秒，均值262.812秒；GPU worker总生命周期24199.408秒，二者边界不同。

|成分|90局总秒|占客户端生命周期|平均秒/局|
|---|---|---|---|
|模型action|4993.070|21.1096%|55.479|
|完整policy减action（video/条件/VAE/host混合）|3759.195|15.8931%|41.769|
|policy RPC其余传输/包装|59.459|0.2514%|0.661|
|cache RPC|643.431|2.7203%|7.149|
|reset RPC|1.115|0.0047%|0.012|
|客户端RPC之外|14196.778|60.0209%|157.742|

RPC之外不能直接称为纯CPU或纯仿真，下面用完整任务profile拆分。reset 90次合计1.115秒、平均12.4ms，占0.0047%，再消除它也无法实质改变总吞吐；full512 BF16 bank已生效。

失败18/90消耗13035.235秒，占55.11%；成功72局平均147.47秒，失败平均724.18秒。微波炉15局仅2局成功，占全部90局耗时48.36%。失败必须保留，不能截断或剔除来提升吞吐。

## 与LIBERO的边界相近对照

相同BAAI A100-SXM4上的既有330条完整独占工程计时：T/C/P客户端总墙钟均值65.08/52.44/57.73秒；本次RoboTwin 90局为270.19/251.22/267.02秒。不同任务、失败率、模型输入和审计方式，不能把比值称为硬件或算法速度比；这里只解释为何体验明显更慢。LIBERO的warm episode均值另为58.28/45.64/50.94秒，不能把它与含初始化的RoboTwin生命周期直接混算。来源LIBERO-COMPARISON.json及原M35 CSV。

完整Target policy：RoboTwin约8.57秒/chunk，LIBERO约3.93秒/chunk。RoboTwin每chunk通常32命令，LIBERO16，不能说每个机器人动作推理必然慢2倍。RoboTwin三路256×320输入、video25、action50、历史window72；当前LIBERO两路128×128、video20/action50、window30。配置改变了工作量，但各项贡献未做单变量归因。

## 一条完整成功局的实际profile

open_microwave/valid16，原Target未装draft Runtime，自然成功；368次提交调用、环境真正推进363次。整局全部命令与未profile的同case完全相同。cProfile总253.605秒；原client状态区间251.806秒；未profile同case238.991秒。插桩约增加5.36%的client墙钟，不能将profile数据用于正式性能或算子对照。

|互不重复的顶层部分|秒|占profile墙钟|
|---|---|---|
|模型policy/reset/cache RPC|102.501|40.42%|
|环境take_action（含双臂规划）|72.469|28.58%|
|动作执行之外的get_obs|46.601|18.38%|
|setup_demo（含planner构建预热）|18.676|7.36%|
|其余导入/配置/命令包装/日志等|13.358|5.27%|

嵌套明细（不再加进上表）：

- 左/右planner各调用363次，累计30.437/30.039秒；每个EE命令分别调用CuRobo MotionGen.plan_single，返回轨迹后再逐物理子步执行。
- take_action内含规划、控制、物理步、状态更新；全进程Sapien step共22132次、6.549秒（含setup），不是一个命令一次物理step。
- get_obs 96次，共47.037秒；其中camera.update_picture本体46.116秒。每次渲染两腕、head和front共四路，而format_obs只发送两腕和head。不能未经验证假定去掉front对其他相机渲染无副作用。
- 场景setup_demo 18.676秒；两臂planner构造/预热17.206秒。每个planner创建并warmup单目标及batch两套MotionGen，评测take_action只调用plan_path。是否能复用/延迟batch需保持solver/RNG/初态契约。
- 5320次torch CUDA synchronize累计39.084秒；其中opt_base.py:172–174由sync_cuda_time触发。包含等待真实GPU规划工作，不能把39秒当纯同步开销直接扣掉。同步后移可能只是把等待搬到result.success.item()/CPU轨迹取回。

## 优化顺序与验收（尚未实施）

1. 只跳过未输入模型、未用于成功判据的front_camera拍摄/取回，保留另外三路原分辨率、rt/32spp/OIDN与拍摄顺序。先同完整局核对三路像素、物理/RNG、完整命令与终态；任何受用图像差异须停止集成，不能以视觉近似宣布等价。收益未知，整条渲染成本46秒只是上限。
2. 让client/planner跨同任务episode常驻，或延迟未调用的batch planner预热；每局仍重建原场景及恢复完整初态/solver采样状态。约19秒初始化是本例可研究上限，不能保证全部消除。
3. 单独研究CuRobo sync_cuda_time是否仅计时用途、能否减少全设备同步；保留相同求解次数/轨迹/物理控制，并测完整take_action，不能只报sync函数变短。不直接替换planner、减少迭代或改仿真频率。
4. 新吞吐稳定后再安排两卡分片；每卡一个现有27GiB worker，同case三臂同卡。模型请求和CuRobo/渲染都使用GPU，不能把RPC之外一概归为CPU不够，更不能假设加CPU核就线性变快。

以上三个方向只属于有证据的建议。没有修改官方renderer/物理/planner源码，没有启动扩大队列；不预先承诺把135 GPU小时降成多少。

## 计数修正与资源

DEV90原actual_commands字段其实是wrapper调用次数。新增DEV90-COMMAND-SEMANTICS.json分列T/C/P提交12224/13376/13408，环境真正推进11808/12927/12958；其差416/449/450是成功或horizon之后的无推进调用。旧数字与原罚分不覆盖；后续质量报告必须标明口径。RoboTwin不能直接套LIBERO800，也不能把提交1520写成物理越过1500。

本次profile GPU生命周期285.933544秒，原M40累计51483.682640/86400秒，原硬截止保留；owned全部退出，server -15/client0。四质量权限false。

原件：DEV90-COST-LEDGER.json、PROFILE-SUMMARY.json、PROFILE-NONOVERLAPPING.json、PROFILE-COMMAND-EQUIVALENCE.json、client.prof、STATUS.json；两条复现见../microwave-reproduction-audit-20261021/RESULT.md。

源码依据（固定版本本地原件）：RoboTwin envs/_base_task.py:437,1479,1574,1629；envs/robot/robot.py:425,459；envs/robot/planner.py:32,74,93；envs/camera/camera.py:273；LingBot evaluation/robotwin/eval_polict_client_openpi.py:423,599；CuRobo opt/opt_base.py:155。
