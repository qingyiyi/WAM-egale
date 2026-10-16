# 投机采样：生产导入路径修复与捕获前准入 CPU 结果

2026-10-12T09:33:45+08:00（Asia/Shanghai）

## 结论

**本步仅CPU修复/验证完成，全部通过；未启动GPU、未训练。下一步是单独确认新GPU预算。**

旧GPU失败是私有guard只认可canonical导入名、误拒生产`modules.model`。本修订放在独立目录，旧job仍为FAIL，旧CPU gate、已执行源码和原始失败全部保留。

## 完成内容

1. 合法路径只允许`modules.model`、`wan_va.modules.model`，同时核对模块文件和spec来源、固定SHA、真实导出class、全部层的原bound forward/代码结构/defaults、原SDPA绑定/代码、eval和原cache；不是任意放宽模块名。
2. `SingleWorkerCHC.admit()`是无前向/不patch/无需KV的只读准入。Runtime在计数wrapper和ready之前检查；capture在fixture读取、reset、preload、infer之前再次检查。优化安装位置仍在capture及历史恢复之后，原capture继续用原attention。
3. 调用真实未mock的生产`load_transformer/from_pretrained`和`_configure_model`，加载临时随机小CPU checkpoint；补同源码双别名、源码/class/forward/backend误配、最后一层和提前拒绝的正负例。
4. 新修订CLI非CPU-probe和Runtime非CPU路径明确锁住，无GPU监督器/排队器/新GPU job；旧GPU标志和旧授权不能启用。

## CPU 结果

| 验证 | 结果 |
|---|---|
| 生产loader正负回归 | **37/37 PASS**：7正例、20非法接口、6前置检查、4CPU边界；含fresh process先导入真实server |
| 原接口回归 | **81/81 PASS**，测试文件原样保留 |
| 生产loader数值夹具 | FP32/BF16小CPU模型，全30层、29变化query的R/C/HC逐bit一致，history/metadata/RNG不变 |
| 同一worker CLI/Runtime probe | 生产`modules.model`真实loader，R/C/HC各29次回放均PASS |
| 顺序验证 | 坏接口在ready/计数wrapper之前拒绝；ready之后被更改的接口也在fixture读取/捕获之前拒绝，0 target前向 |
| 边界 | 0 CUDA初始化、0新GPU forward/job/排队、0episode/EGL/完整chunk/训练 |

CPU-001首次通过；三阶段总 **113.305秒（约1分53秒，含子进程导入）**。其中37项测试本体46.561秒、81项28.091秒；这不是GPU性能计时。

## 不变与未证明

- 269份前置/旧job源码与证据SHA不变，包含197前置文件和15旧私有执行文件；旧GPU FAIL不改。受保护state/rules不改、cycle仍不存在，9份target文件stat、旧fixture及A_off保持不变。
- 公共模型/server/batch14/attention不改。`capture.py/contract.py/test_cpu.py/gpu_compare.py`逐字一致；adapter除install准入及新增admit外的旧method AST不变；Runtime除constructor/capture准入及新增check_interface外的旧method AST不变。历史KV复用、失效、数值数学、原SDPA/cat和成本计时方法未重构。
- 这是随机小checkpoint的生产**函数/API路径**验证，不是完整真实target GPU constructor/模型数值/显存/成本验收，也不证明成功率、环境步数或E2E。未做独立审计。
- 仅投机采样任务，HW增量追加保留TILE既有记录，不修改TILE算法或把其收益混入本路线。

## 下一步与目标

下一步单独确认一次全层C/HC真实GPU验证预算：建议仍1 worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练。确认后才补新修订的默认关闭GPU入口及新job预检，不复用旧FAIL job或旧CPU gate；接口前置准入→恢复/捕获→数值/失效/隔离→六块干净OFF→独立ON→清理，失败停不补跑。

目标：真正验证长历史单worker的全层C/HC在不扰动输出/状态、显存可承受条件下，是否降低**含冷cache构建和释放的完整verifier成本**。有净收益再讨论完整policy/冻结target闭环，不把CPU结果当GPU速度。

[修订说明](../../../投机采样/experiments/single_worker_chc_loader_fix/README.md)；[CPU结果](../../../review/OMISSIONS.md)；[37项](../../../review/OMISSIONS.md)；[81项](../../../review/OMISSIONS.md)；[同worker probe](../../../review/OMISSIONS.md)；[不变核查](../../../review/OMISSIONS.md)。
