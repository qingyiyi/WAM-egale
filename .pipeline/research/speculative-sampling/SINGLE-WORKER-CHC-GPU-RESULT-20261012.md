# 投机采样：单worker全层C/HC GPU结果与CPU复盘

2026-10-12T09:06:11+08:00（Asia/Shanghai）

## 结论

**本次GPU job失败，已停止并释放本任务GPU；没有新的C/HC速度结论。**

- 新私有监督器已补齐，CPU关通过32项任务回归、81项canonical真实接口回归及同worker全30层CPU probe；这些测试没有覆盖生产loader的导入别名。这是本次包装与预检的覆盖缺口。
- 稳定空闲资源关通过后只启动1个模型worker。GPU加载至清理 **75.812秒**；target预留/尝试/完成 **52/52/52**（计划895/硬限900），0完整episode/EGL/完整policy chunk/environment step/训练。
- capture执行到首candidate历史恢复后的包装安装；此前29变化query和30层/2144历史几何检查在执行路径中已通过，但capture整关**未通过**，RAM query/output本体未形成持久化合格产物。
- 包装安装失败后没有进入R资源probe、C/HC等价性、十二类失效、13行隔离、warmup、六块干净OFF或ON诊断。不能由日志的原始capture耗时推算C/HC净收益，也不能推出方案无效。

## 根因：私有包装误拒合法生产导入

服务器通过 `from modules.utils import load_transformer` 加载 `modules.model.WanAttention`；CPU夹具通过 `wan_va.modules.model` 构造模型。两者指向同一份固定源码，但在Python中是不同模块/类/函数对象。私有adapter只接受后者，触发 `LeaseError('requires unmodified eval original WanAttention interface')`。

CPU复盘通过 **58项检查**：使用真实服务器的loader类和`_configure_model`创建小几何随机CPU全30层模型，无checkpoint/模型前向/CUDA；精确复现错误。生产30层只有模块名条件失败，原class forward、源码路径、eval、cache=original和各自原SDPA均通过；canonical模型安装/撤销仍通过。**这是我的包装/测试覆盖问题，不是target精度、训练模式、attention数值或性能失败。**

## 清理、边界与HW

- 归属清理PASS：本任务worker/supervisor退出，没有本任务GPU PID或端口残留；不信号其他进程。清理时NVML allocated=0、compute/graphics为空。随后服务器GPU被其他用户任务使用，与本job无关；本job不排队/重试。
- 197份前置冻结文件、15份本次私有实现、A_off、fixture统计、9份target文件统计及受保护state/rules/cycle缺席状态保持不变。旧FAIL、CPU gate和执行实现不改写。
- AUTH、CPU、START、FAILED已逐关写入HW；4类index每类原65条完整保留，新增4条至69条；含TILE的旧记录不丢。本CPU复盘另增1条，仍明确非TILE。
- 这是一次按授权失败停止的验证结果，不把CPU58项诊断写成GPU验收通过；不训练、不RL/联训、不mega Transformer。

## 下一步

下一步先确认仅CPU修复私有包装的源码校验导入别名、补真实loader路径的正/负回归，并把接口准入移到52次捕获之前。公共源码/batch14/原SDPA/冻结权重不改；CPU通过后另确认新job GPU预算，本次不修复补跑、不训练。

目标：在真实生产类上保留全部保护、尽早拦截接口问题，再验证完整verifier是否逐bit不扰动、额外显存是否可承受，以及含冷cache构建的净收益；随后才讨论完整单worker policy/闭环和环境步数。

[原始简报](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/BRIEF.md)；[GPU失败](../../../投机采样/results/runtime/p0-single-worker-chc-20261012-085017/FAILED.json)；[CPU58项诊断](../../../review/OMISSIONS.md)；[复现脚本](../../../review/OMISSIONS.md)。
