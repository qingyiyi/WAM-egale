# AutoDL TF23 研究质量评测：独立源码与准入

本目录只准备独立研究评测，未部署、未运行。旧 `autodl_postquality_two_stage_v1`、旧 watcher、原 r2 `FAILED`、已冻结的 parent10 与模型权重均不改。真实新 GPU 作业的前置是旧100完整终态并归档清理，及同 seed TF23 parent2000/child2500 两个可比 DEV 终态完整。当前 parent10 未完成时，只能取得 `WAIT_OR_FAIL_RESEARCH_DEV_SELECTION`；不能生成可部署候选。

选择规则在 `local_gate.py` 固定：10个相同 task/init 上先比成功数，再比所有10局的失败按800步计的平均分，并列选 parent2000。原 r2 因相机像素差异仍是 `FAILED`，独立诊断证明物理/控制器/环境与实际 target RNG 配对时才允许**研究测量**；`quality_gate_passed` 单独保存原 DEV 工程门真假，不将候选称为胜者或证明3pp非劣。`DEPLOYMENT-SELECTION.json` 只是研究部署视图；原训练 `selected.json`、parent/child checkpoint 的 schema、权重、归一化和元数据字节不改。parent 调用冻结 stage1 loader，child 调用冻结 stage2 loader。两者都先过完整20237条实际 CAL 的同 CPU 逐位接口门，再由 `QUALITY_DRAFT` 在同 GPU 跑完整20237条实际查询逐位门；这些门不证明闭环成功率。

远端顺序：旧100完整归档 → 源码与紧凑 CAL20 部署 → **原子写入一次** `BUDGET-ORIGIN.json` → 脱离 SSH 的 CPU 父时钟串行监督原 T5 的11 prompt、7核 bank、104计划准备、完整20237 CPU CAL → 4条完整 T/A_OFF 原编码/缓存严格等价 → 新 T50/所选 TF23 50。Long10 task0–9/init25–29、原 horizon800、K2/B2 bonus、原阈值和选择器不变。旧50 T 不复用。每条完整原始命令、初末状态、两个真实初始相机像素、RNG、请求与保存张量覆盖均保留；T/TF 物理控制器和实际 target RNG 严格配对，相机差异逐像素报告而不设事后容差，不宣称差异局精确同输入。成功率、逐 task/init 四格、Wilson 描述区间、固定 Long10 内成对 init bootstrap、失败800罚分、仅双方成功的步数比、target/chunk 与接受深度分别报告。

同一四小时累计预算从**远端 CPU bank 之前**计，包含 bank、计划、CPU/GPU CAL、104局、分析和收尾；300秒留给归属进程清理，另受北京时间2026-10-16 10:30硬截止约束。AutoDL 实查 cgroup 为10核配额，bank和目标server用7线程，EGL client及完整CPU CAL用2线程；bank自身最多3600秒，仍受共享截止提前截断。CPU 父时钟及 GPU 父时钟均独立于 SSH/NVML，记录 PID、start_ticks、UID、argv，并只清理已证明归属的子树。中途失败优先于 PARTIAL；局数不足104只能签 PARTIAL/FAILED，不能缩短 horizon 或隐去未完成局。当前本机 r2 的 T10 平均136.89秒、TF10平均103.52秒，100局纯 episode 约3.34小时；另有远端 bank 约30–45分钟、重置、CAL、分析，所以四小时**不保证**收齐104局。共享 AutoDL wall 只供诊断，正式加速比仍待本机独占窗口。

全部前置终态就绪且审阅源码后，本机 CPU 准备命令如下；`--out` 必须是新路径，当前不可提前生成准入包：

```bash
cd /rjs/cxz/论文探索
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /rjs/cxz/miniconda3/envs/va/bin/python -B \
  投机采样/experiments/autodl_research_quality_v1/prepare_bundle.py \
  --out 投机采样/results/runtime/p0-autodl-research-quality-preparation-v1-20261015-r1/bundle-r1
```

审阅 `BUNDLE-MANIFEST.json`、`handoff/local-dev/SELECTION.json` 与完整 parent10/旧100 终态后，唯一显式 dispatcher 命令是：

```bash
cd /rjs/cxz/论文探索
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /rjs/cxz/miniconda3/envs/va/bin/python -B \
  投机采样/experiments/autodl_research_quality_dispatch_v1/watch.py \
  --bundle 投机采样/results/runtime/p0-autodl-research-quality-preparation-v1-20261015-r1/bundle-r1 \
  --out 投机采样/results/runtime/p0-autodl-research-quality-dispatch-v1-20261015-r1 \
  --execute-reviewed
```

`prepare_bundle.py` 只作本机 CPU 文件准备；`watch.py` 才有远端写入和启动能力。dispatcher 先只读等旧100并真正归档，再上传独立 bundle，绝不修改旧100热源码/预算。它只在远端 CPU 父时钟三个阶段全部 PASS、清理全部通过及 CPU 全 CAL receipt 绑定所选权重后，启动一个 GPU deadline wrapper；没有自动重试、续租、预算重置或未完成局补跑。远端 source/config 与全部模型接口做集成来源核验，不运行额外 standalone SHA job，也不复制大 target 权重。
