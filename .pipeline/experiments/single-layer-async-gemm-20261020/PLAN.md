---
kind: experiment
id: M39
cycle: wam-research-20261016
status: registered
---
# 一条SM80 GEMM双缓冲流水路线

用户已明确继续优化；M38支持先处理GEMM。本轮一条实现路线，不搜索tile/grid/seed。原whole_layer.cu保持；新算法为optimized/csrc/gemm_pipeline.cuh，由独立构建入口生成新namespace/DSO，避免影响baseline或源码带读。

变化：64×64×32、相同K顺序和WMMA累加/BF16 bias边界；LD40共享布局已在M34有限验证。新版本用两个shared buffer，warp1/2使用SM80 cp.async加载下一块，warp3–6算当前块，wait_group完成后CTA barrier交接，保留warp0领取与warp7写回。没有跨stage流水，也不改attention/缓存协议。

对照：原whole、相同LD40的同步版本、LD40双缓冲版本；native保留成熟库且同输入核对Torch。同步LD40是因果对照，不能把padding收益全归为异步流水。另有唯一双缓冲阶段计时副本，profile与正式样本分离。

输入：既有9真实block0，B2/B14 action Q16及B2 video Q128，各短/中/长历史。gate逐项比较原whole/同步/异步全部中间量、输出、有效只读KV、tile exactly-once、输入/参数/RNG；原Torch偏差继续记录。另做B2 action/video共6起点的update1→2→0合法缓存序列，比较旧whole与新异步，不继承旧写入精度证书。

只有异步/同步保持失败whole的有限算术与状态时才采性能。两个独立进程，每签名每路径一次预热、六次交错完整block计时；缓存恢复和profile在计时外。每个签名记录实际kernel数量与资源。原Torch逐位FAIL时仍属未准入原型性能，不能接入投机或声称闭环加速。

远端GPU0一worker、CPU4/5两核；GPU生命周期总帽3600秒（含构建、等待、失败、清理），T0+3300停止工作、+3600清理。合计完整block≤900、0独立采样/episode/30层/训练，新增≤512MiB。构建≤1200秒；gate≤600秒；两个性能worker各≤600秒；阶段profile≤300秒。每阶段启动前检查剩余生命周期及其他GPU context；不碰别人进程。

若编译/接口有确定实施问题，保留attempt后可在原截止/总预算内做至多一次具体修复，不自动更换研究配置或扫参。数值/状态不一致则不计性能，通过前不推广。清理以PID+birth/PGID/UID核对并记录真实退出。

交付：源代码、源码/输入manifest、9点与写序列门、两进程计时、profile、资源终态、HW/main小证据。四false不变；没有permission提升。
