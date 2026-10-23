# CUDA Graph 比较条件澄清

来源：用户提出“guard不可以用cuda graph，因为我们自己实现的也没有使用cuda graph”；本次只核对已有源码和计时条件，没有启动补测或改变原始结果。

## 已观察事实

本轮 T4 的 A、L1、L13、D、G **全部使用 warm CUDA Graph**。自有实现的算子内部是一个 persistent kernel；测试脚本在外部捕获其 forward，随后重放。算子本身不依赖 Graph，与计时脚本使用 Graph 可以同时成立。

证据：[T4-final1/timing.py](T4-final1/timing.py) 第56–66行对每个 arm 创建 Graph，第64–65行捕获 Torch block 或自有 ext.forward，第67–76行对所有 arm 预热并计时 replay。第二独立进程采用同样路径。附件第5.4节要求无插桩、完整单层 warm CUDA Graph。

|计时条件|自有 A/L1/L13|Torch D / guarded G|本轮状态|
|---|---|---|---|
|warm Graph|Graph 捕获与重放|Graph 捕获与重放|已完成，见最终同轮表|
|普通 eager 调用|关闭 Graph|关闭 Graph|NOT_MEASURED，不能借用上行数字|

## 判断与后续建议

公平比较应明确双方的 Graph、缓存、输入与计时边界。不能只给一方开启 Graph，也不能把“源码没有 Graph”误写成“本次测试没有 Graph”。Graph 重放不会把 guarded 的多个 kernel 自动变成一个 persistent kernel；它改变的是提交执行方式。

若研究目标包含普通调用下减少 launch 的收益，可单列双方都关闭 Graph 的对照；这属于新增测量，当前没有启动。保留现有双方 Graph 结果，分别回答优化执行条件下的完整层成本与普通提交路径成本，不混算加速比。用户当前的疑问不记成“只有 guarded 使用 Graph”的事实，也不自动改写已完成实验的协议或预算。
