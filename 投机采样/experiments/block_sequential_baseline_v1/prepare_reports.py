"""Build the bounded CPU evidence packet from actual attempt results only."""
import argparse
import ast
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path

REVIEW_COMMIT = "ca5f37e0482c2e38e3738e3d8e71e3b185990a23"
FLAGS = dict(engineering_quality_screen=False, independent_test_complete=False,
             noninferiority_established=False, formal_performance_permission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / ".pipeline/experiments/wam-block-baseline-20261019"
    code = root / "投机采样/experiments/block_sequential_baseline_v1"
    def read(path):
        return json.loads((report / path).read_text())
    def write(name, content):
        (report / name).write_text(content.rstrip() + "\n")
    def write_json(name, data):
        write(name, json.dumps(data, ensure_ascii=False, indent=2))
    cpu = read("attempts/cpu-real-block-final/CPU-ACCURACY-RESULT.json")
    stage = read("attempts/cpu-stage-layout-final/CPU-STAGE-AND-SHAPE-RESULT.json")
    parameters = read("attempts/cpu-real-block-final/BLOCK-PARAMETERS.json")
    fixtures = read("attempts/cpu-real-block-final/FIXTURE-MANIFEST.json")
    provenance = read("SOURCE-MANIFEST.json")
    noise = read("OLD-PERFORMANCE-NOISE-AUDIT.json")
    if not cpu["collection_complete"] or not cpu["status"].startswith("COMPLETE_CPU_LOGICAL_CONTRACT_PASS"):
        raise RuntimeError("Do not manufacture reports before the CPU replay terminal")
    if stage["status"] != "COMPLETE_CPU_STAGE_AND_STATIC_REPLAY_PASS_GPU_PENDING":
        raise RuntimeError("Stage evidence is not complete")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    source_count = sum(entry["path"].startswith("投机采样/") for entry in provenance["source_records"])
    source_entries = []
    patch = []
    for path in sorted(list(code.glob("*.py")) + [code / "README.md", code / "LICENSE.upstream.txt"]):
        raw = path.read_bytes()
        relative = str(path.relative_to(root))
        entry = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if path.suffix == ".py":
            entry["symbols"] = {node.name: dict(line=node.lineno, end_line=node.end_lineno)
                                for node in ast.parse(raw.decode()).body
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        source_entries.append(entry)
        patch.extend(difflib.unified_diff([], raw.decode().splitlines(keepends=True),
                                         fromfile="/dev/null", tofile=relative))
    patch_bytes = "".join(patch).encode()
    (report / "B0-SOURCE-PATCH.diff").write_bytes(patch_bytes)
    implementation = dict(base_commit=REVIEW_COMMIT, scope="independent new namespace only",
                          patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
                          source_entries=source_entries, model_tensor_hash_performed=False)
    write_json("IMPLEMENTATION-MANIFEST.json", implementation)
    key = {row["case_id"]: row for row in cpu["rows"] if row["update_cache"] == 0}
    schema = dict(schema="wam-block-contract/v1", recorded_at=now, review_commit=REVIEW_COMMIT,
                  implementation_manifest="IMPLEMENTATION-MANIFEST.json",
                  reference=provenance["source_records"][0],
                  implementation="B0_HYBRID_SEQUENTIAL_TORCH_PRIMITIVES",
                  production_integrated=False, GPU_jobs=0, scopes=["CPU_SERIALIZED_LOGICAL_ACTION_BLOCK0"],
                  architecture=dict(hidden=3072, heads=24, head_dim=128, FFN=14336,
                                    context_tokens=512, action_query_tokens=16,
                                    num_model_blocks=30, original_norm_eps=1e-6,
                                    activation="GELU approximate=tanh",
                                    parameter_count=parameters["parameter_count"],
                                    parameter_bytes=parameters["parameter_bytes"]),
                  weights=parameters, fixtures=fixtures,
                  stage_observations=stage["rows"],
                  backend_observations=stage["backend_observations"],
                  original_CUDA_layout="UNKNOWN_NOT_CAPTURED; no equality claim",
                  original_rng="UNKNOWN_NOT_CAPTURED_IN_COMPACT_FIXTURES",
                  original_sigma_timestep="UNKNOWN_RAW_VALUES; exact temb remains in referenced original PT",
                  actual_CUDA_GEMM_dispatch="UNKNOWN_NO_GPU_RUN",
                  alias_contract="memo preserves repeated serialized objects; production alias topology unknown",
                  cache=dict(engine="original mask/index", readonly_B14="no cache writes",
                             readonly_B2="temporary slots then restore mask; invalid metadata may change as in R0",
                             state_equality="all valid KV finite/equal; all post metadata equal; input/RNG unchanged",
                             writes="same saved B2 query under original update_cache=1/2; counterfactual state controls, not robot samples",
                             invalid_KV="excluded from semantic comparisons; never claim initialized padding identity"),
                  quality_permissions=FLAGS)
    write_json("block_contract.json", schema)
    accuracy = dict(schema="wam-block-accuracy-summary/v1", recorded_at=now,
                    review_commit=REVIEW_COMMIT, implementation_manifest="IMPLEMENTATION-MANIFEST.json",
                    status="CPU_FINITE_BITWISE_STATE_PASS_GPU_PENDING",
                    scope="CPU normalized serialized real action block0; not original CUDA execution",
                    collection_complete=cpu["collection_complete"], finite_pass=True, bitwise_pass=True,
                    tolerance_pass=True, state_contract_pass=True,
                    core_fixtures=6, derived_write_controls=6, unique_real_fixtures=6,
                    execution_count_current_attempt=36,
                    CPU_contract_and_negative_tests=22,
                    GPU_driver_prepared=True, GPU_driver_execution="NOT_RUN",
                    CPU_result="attempts/cpu-real-block-final/CPU-ACCURACY-RESULT.json",
                    CPU_stage_result="attempts/cpu-stage-layout-final/CPU-STAGE-AND-SHAPE-RESULT.json",
                    GPU=dict(collection_complete=False, finite_pass=None, bitwise_pass=None,
                             tolerance_pass=None, state_contract_pass=None,
                             status="NOT_RUN_NO_GPU_AUTHORIZATION"),
                    performance_status="NOT_RUN_NO_GPU_AUTHORIZATION", **FLAGS)
    write_json("ACCURACY-RESULT.json", accuracy)
    table = "\n".join(
        f"| {case} | 0 | {row['complete_block']['bitwise_pass']} | {row['post_state']['pass_all']} | "
        f"{row['historical_gpu_gold_cpu_comparison']['bitwise_pass']} | {row['historical_gpu_gold_cpu_comparison']['max_abs']} |"
        for case, row in key.items())
    write("FACTS-AND-UNKNOWNS.md", f"""# 完整block基线：事实与未知

生成时间：{now}（北京）。科研base：`{REVIEW_COMMIT}`。本轮为CPU准备，不启动新GPU。

OBSERVED：已有v2只替换60个self/cross SDPA入口，投影/norm/FFN仍为原Torch。固定提交包含108 attention、54 block、六30层query、三短中长冻结policy的有限GPU数值证据；这不是全自有设备block或机器人闭环证书。

OBSERVED：现在已新增可单独调用的B0_HYBRID_SEQUENTIAL，真实参数与显式self→cross→FFN阶段；没有调用原block.forward代理完成B0。最终版本六个真实输入+六个同输入合法写控制，共36次CPU完整block执行，R0/R0自复现、R0/B0输出逐位、有效KV有限及状态/输入/RNG通过。重复核验不增加独立样本数。

| 真实fixture | update_cache | CPU R0/B0字节相同 | 有效状态通过 | CPU R0/旧GPU gold字节相同 | CPU/旧GPU max_abs |
|---|---:|---|---|---|---:|
{table}

最后两列保留不同设备数值环境的真实差异，不能拿CPU通过覆盖GPU待验。CPU结果主张只涉及本轮相同CPU环境的R0与B0。

OBSERVED：最终源的short-B2、long-B14各28个可观察公共边界通过，静态原形状context K/V投影的R0/B0控制均通过CPU。新的静态block适配器仅供固定输入诊断，不提供生产缓存失效协议。

OBSERVED：22项CPU契约/负例通过，包括逻辑字节与非有限数、每臂模块/cache独立且同原Parameter对象共享、无GPU授权拒绝、模拟Event的状态恢复及ABBA/BAAB。模拟Event只检验计时协议，不是延迟测量。完整GPU driver已准备，尚无GPU执行证书。

OBSERVED：读取固定提交{source_count}个小源与16份既有结果，均存在；本地存在的相应文件逐字一致。未做权重/tensor hash。参数为原checkpoint block0的27项，共{parameters['parameter_count']:,}参数、{parameters['parameter_bytes']:,}字节，只加载单block到CPU。

UNKNOWN：B0在GPU的数值、状态、延迟、峰值显存、CUDA实际GEMM dispatch、完整模型及policy集成；原生产CUDA stride/storage offset/alias及compact中缺失的原RNG、原始sigma/timestep数值（仅精确temb由原PT保留）；其余层/时间段、完整video block的新B0证书。不得补造。

HYPOTHESIS：较大的候选行数可能帮助后续设备调度，尚无本轮GPU因果证据。旧计时波动的CPU/host/设备根因未分离。

RECOMMENDATION：下一步唯一优先是新授权下的六fixture GPU完整block数值/状态→A/A→匹配配对计时；不是马上开发persistent scheduler。四项机器人质量/正式性能权限仍false。
""")
    write("BASELINE-COVERAGE.md", """# 替换范围与验证范围

| 实际运算 | R0 | 已有R1 | 新B0默认 | 设备tile状态 |
|---|---|---|---|---|
| self SDPA | TORCH_ORIGINAL | OWN_STANDALONE | TORCH_ORIGINAL，可另挂aligned但本轮GPU未运行 | 现有独立attention kernel，不等于完整DAG内可调用原语 |
| cross SDPA | TORCH_ORIGINAL | OWN_STANDALONE | TORCH_ORIGINAL，可另挂aligned但本轮GPU未运行 | 同上 |
| self/cross QKV和输出投影 | TORCH_ORIGINAL | TORCH_ORIGINAL | TORCH_ORIGINAL，由新Python阶段组织 | GEMM仅host调用，没有新D0 |
| Q/K RMSNorm、FP64 complex RoPE | TORCH_ORIGINAL | TORCH_ORIGINAL | TORCH_ORIGINAL运算/边界 | 无新设备原语 |
| norm、调制、gate、残差、cast | TORCH_ORIGINAL | TORCH_ORIGINAL | 原Torch数学，MIXED控制流 | 无新设备原语 |
| FFN1、GELU(tanh)、FFN2 | TORCH_ORIGINAL | TORCH_ORIGINAL | 原Torch模块/functional，显式边界 | 无新设备GEMM链 |
| mask/index KV和metadata | TORCH_ORIGINAL | TORCH_ORIGINAL | 原helper与原分支组织 | 有效语义与写状态单独核对 |
| shared/guarded候选历史 | 原B14 dense materialization | 既有独立实验 | 独立adapter准备，GPU NOT_RUN | 原kernel有有限旧证据，联合新基线不能借证书 |
| context投影缓存 | 原core未缓存 | 原core未缓存 | 两个真实fixture静态CPU控制 | 普通缓存，非MegaTransformer新贡献 |

| 实现 | kernel/子层 | 完整block | 完整30层 | 完整policy | 机器人闭环 |
|---|---|---|---|---|---|
| 已有R1 v2 | 108有限GPU attention；九保存输入 | 54有限GPU点，逐位/有效状态 | 六冻结query有限通过 | 三短中长冻结配对通过 | 本轮未验证；四false |
| 新B0_TORCH | 两fixture×28可观察阶段CPU | 六真实action输入+六合法写CPU控制通过 | NOT_RUN | NOT_RUN | NOT_RUN |
| B0_ALIGNED | 接口已准备 | GPU NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 新联合R_strong适配 | 旧shared/guarded范围可查；本轮native未导入 | 联合GPU门未运行 | NOT_RUN | NOT_RUN | NOT_RUN |
| D0/P_seq/P_sched | NOT_IMPLEMENTED | NOT_IMPLEMENTED | NOT_IMPLEMENTED | NOT_APPLICABLE | NOT_APPLICABLE |

已有R1替换范围只为SDPA；验证到完整policy不扩大其实现所有权。新B0是完整顺序host基线，不能叫全自有CUDA block。collection_complete/finite/bitwise/tolerance/state/performance分字段见ACCURACY-RESULT.json。

支持证书：layer0/first、B2或CFG2×7=B14、action Q16、K288/1152/2160、hidden3072/H24/D128/FFN14336、BF16参数/主激活、指定FP32/FP64边界、original cache。三份video保存文件没有block_args与完整cache metadata，完整video B0为BLOCKED_MISSING_FIXTURE；旧video attention证据仍保留。不是随机替代或重新采集。
""")
    write("BLOCK-CONTRACT.md", f"""# 真实block契约与依赖

科研base `{REVIEW_COMMIT}`；完整机器清单见[block_contract.json](block_contract.json)、参数来源见`attempts/cpu-real-block-final/BLOCK-PARAMETERS.json`、源hash/函数行号见[SOURCE-MANIFEST](SOURCE-MANIFEST.json)。

OBSERVED：WanTransformerBlock，dim3072、H24、head128、FFN14336，cross_attn_norm=true、eps=1e-6。27项参数全部BF16；原config attn_mode=flex，但真实server显式以LINGBOT_ATTN_MODE默认torch覆盖（wan_va_server.py:264–275），本轮AST原类也只准入torch分支。未导入整个server/编译Flex/加载全模型。

R0直接调用固定AST原forward。B0复制原运算组织、共享同一参数对象、eval/inference，autocast关闭；norm1/3为非affine FP32LayerNorm，norm2为affine FP32LayerNorm；QK为跨3072维RMSNorm后unflatten为H24/D128。FFN是Linear→GELU(approximate=tanh)→Dropout(0)→Linear。

| Linear | B2逻辑M | B14逻辑M | N | K |
|---|---:|---:|---:|---:|
| self Q/K/V/out，cross Q/out | 32 | 224 | 3072 | 3072 |
| cross context K/V | 1024 | 7168 | 3072 | 3072 |
| FFN1 | 32 | 224 | 14336 | 3072 |
| FFN2 | 32 | 224 | 3072 | 14336 |

以上是实际CPU模块输入/权重几何；观察到可零copy view为[M,K]。保存后的主输入stride为[B×Q,Q,D]对应的连续stride，线性weight为[N,K]连续；leading row stride为K（FFN2为14336）。**实际CUDA GEMM调用、kernel和tile仍UNKNOWN**；CPU前端观察不替代GPU profile。当前ATen mode未观察到内部addmm/bmm，不补造派发证据。

CFG行映射是[0×7,1×7]。每个candidate独立Q16、私有16个K/V，历史来自对应CFG的B2有效prefix272/1136/2144；不能描述成跨候选Q112/224 attention序列。B0原B14仍执行history gather/index_select/cat，不声称共享prefix优化或只读HBM一次。context为[batch,512,3072]，原block无cross投影缓存；静态适配只固定同一context对象与原shape。

显式边界：temb和调制FP32；norm1/2/3先FP32，调制后回BF16；self residual先FP32再BF16；cross residual按原BF16相加；FFN projection与GELU之间保留BF16舍入；FFN final gate/residual先FP32再BF16。RoPE保持原FP64→complex乘法→BF16，freq输入complex64。不更换激活/norm/归约顺序。

```mermaid
flowchart LR
  H[hidden BF16] --> N1[完整3072维LN与时间调制]
  N1 --> QKV[独立QKV GEMM]
  QKV --> QR[QK完整行RMSNorm与原RoPE]
  KV[有效历史KV与候选私有边界] --> SA[self attention可见KV归约]
  QR --> SA
  SA --> PO[输出投影与self gate/residual]
  PO --> N2[FP32 cross LN再BF16]
  N2 --> CQ[cross Q]
  CT[512 context原K/V投影或固定缓存] --> CA[cross attention]
  CQ --> CA
  CA --> CR[输出投影与BF16 residual]
  CR --> N3[完整3072维LN与调制]
  N3 --> F1[FFN1 14336维]
  F1 --> G[原BF16 GELU tanh]
  G --> F2[FFN2 归约完整14336维]
  F2 --> O[FP32 gate与residual再BF16]
```

norm/RMSNorm消费者需完整行归约完成；attention需对应candidate/head完整合法可见KV；输出投影需其K维完整归约；FFN2某行最终结果需14336维activation reduction，不能FFN1任意N tile刚完成就发布最终ready。流水可研究行组/中间tile，但尚无device实现、同步或收益证明。

生命周期：原权重只读全程共享；六个时间调制分量view同一table；QKV与RoPE临时值服务本次attention，private不写B14历史；self/cross residual用完即释放额外Python引用；FFN1 expansion在GELU完成后释放，GELU/Dropout结果在FFN2后释放。debug tap保存快照会延长生命周期，只用于诊断，计时必须关闭。

只读B14应保持全部metadata/有效KV；B2 update0按原helper暂写无效槽并restore mask，无效槽id/is_pred可能合法变动，不当作有效KV污染。update1/2比较完整最终metadata和有效KV，不要求写后不变。原helper在空间不足时会驱逐，不能从当前有空槽的六fixture推广为所有容量下只读无副作用；驱逐只在标明SYNTHETIC_ONLY分支控制中比较等价。

原捕获`.cpu().contiguous()`已丢原CUDA layout/alias；本轮不声称恢复它。可变cache每臂独立，read-only参数共享；当前CPU RNG逐位恢复/保持不代表原采集时CUDA RNG已保存。所有这些缺口保留在JSON。

参考方法：[PyTorch2.9数值说明](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)说明设备/批处理数值可不同；[Linear](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html)与[LayerNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html)仅支持API解释，不替代本地执行证据。
""")
    write("PERFORMANCE-LIMITATIONS.md", f"""# 性能：本轮没有新GPU时间

PERFORMANCE-RESULT.csv的54行都是NOT_RUN_NO_GPU_AUTHORIZATION，数值为空，不是0ms。benchmark_block/gpu_matrix/gpu_job提供已绑定的预算驱动、A/A、ABBA/BAAB与准备矩阵；本轮只做CPU控制/负例测试，完整GPU执行尚未发生，必须先有新授权、资源准入与fixture/state证书，不能声称已经正式benchmark。

已有R1的54点完整block配对比中位{noise['R1_over_R0']['median']:.6f}、几何均值{noise['R1_over_R0']['geomean']:.6f}，范围{noise['R1_over_R0']['min']:.6f}–{noise['R1_over_R0']['max']:.6f}，9点>1.05；六30层中两例>1.05。只是旧有限测量，不是B0结果。

重新读取固定CSV发现，54个原Torch block每点三轮“轮内中位数”的max/min有45点>1.05、36点>1.10，中位1.172230，最大7.387167。没有独立A/A控制；三轮只有三个执行块，不把每个Graph replay/元素当独立样本。此观察不证明是CPU、GPU、时钟或某个算子根因；不能据此删除旧慢点或挑最快轮。

新方案：先A/A，若三轮配对噪声不能分辨5%门则PERF_PENDING；A/A建议偏差上限2.5%只是新工程准入。再同cache、同eager边界交错R0/R1/B0/合格R_strong，至少三轮，保存全部原始Event和同步wall。prepared execution与含状态恢复/布局/缓存准备的integration时间分开；冷构建另记；正常计时无hook/profiler/逐算子同步。事件区间会包含host launch gap，不冒充纯kernel活跃时间。

完整block Graph原路径存在动态nonzero/tensor-to-Python缓存分支，当前NOT_IMPLEMENTED，不强行capture/跳过CPU更新；已有attention Graph-core与block eager不能相除归因。guarded/shared/static投影组合先逐位/状态验证，不能按名字拼强baseline；既有OriginalShapePack要求真实30层，新的block-only静态诊断不伪造30层对象。

未来B0工程筛查：六核心fixture配对比几何均值≤1.05，无可复现分层>1.10；噪声大/资源异常则UNKNOWN而非PASS。不是旧质量门或统计非劣标准。block层没有g/round提交，不报告ΣT_round/Σg，更不能给机器人E2E加速。

计时依据：[PyTorch2.9 CUDA语义](https://docs.pytorch.org/docs/2.9/notes/cuda.html)和[CUDA12.4.1 Best Practices](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html)。原报告与CSV见SOURCE-MANIFEST，旧失败不改。
""")
    write("SCHEDULER-READINESS.md", """# 是否可开始tile scheduler

结论：**尚不可开始收益验证型persistent scheduler**。完整host顺序B0的CPU基线已建立；GPU数值、状态、A/A环境与强baseline配对时间尚缺。

| 项目 | 当前状态 | 缺口 |
|---|---|---|
| 原SDPA的独立CUDA kernel | EXISTS，旧有限GPU通过 | host入口含ATen分配/launch；vendor存在CTA绑定的device模板，但未形成调度器可组合ABI/资源证书 |
| shared/guarded历史attention | EXISTS，有CTA绑定device模板及有限旧输入通过 | 固定128线程/extern SMEM/同步；新联合cache/block仍须门，非可随意调用的全block原语 |
| GEMM | 原Torch/cuBLAS host接口 | 无已对齐且不明显慢的D0设备tile primitive |
| norm/GELU/row ops | 原Torch host接口 | 无新设备调用约定、资源/舍入证书 |
| P_seq/P_sched | NOT_IMPLEMENTED | 同一D0、同数学配置后才比较调度增量 |
| 完整block调度DAG | CPU阶段已列依赖 | 无真实GPU profile占比、ready粒度/同步/寄存器/SMEM测量 |

唯一下一优先：先让新B0在六真实fixture通过GPU数值/有效状态门与A/A/完整block配对计时；噪声失败先解决可比较性。通过后再只对实际热点选择一个D0，例如FFN1→BF16 GELU→FFN2行组链；目前只是备选，未按FLOPs猜成最大时间热点。

条件触发设计：静态任务结构、轻量ready计数、CTA领取M/N输出tile；第一版不加跨CTA split-K，必要global scratch可保留。LayerNorm需完整hidden行，FFN2需完整activation K reduction，必须保留BF16中间舍入。更多candidate行不自动提供任意跨层/跨token独立性；A100没有Hopper TMA/WGMMA。

潜在收益上界只能用实测份额f算Amdahl，例如理想上界1/(1-f)；当前f未知，不给虚构倍数。重写GEMM损失权重复用、Tensor Core效率或改变累加顺序可吞掉launch收益。P_sched>P_seq才支持调度增量；仍慢于R_strong不能宣称系统胜出。

借鉴[CUTLASS4.2.1 grouped scheduler](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/grouped_scheduler.html)仅指任务领取设计；它不是完整WAM依赖DAG实现。本轮无D0/P_seq/P_sched代码、无30层fusion、无GPU调度实验。
""")
    claim_rows = [
        ("当前独立实现只接管SDPA", "固定integration/install与CUDA launch", "已有源码与有限R1证据", "其余子层仍Torch", "允许限定描述"),
        ("B0已显式执行完整block", "自身阶段传递、真实参数、完整输出/状态", "六真实CPU fixture+六合法写控制通过", "GPU/其余层/原layout未验", "仅CPU保存后契约"),
        ("逻辑M由32变224", "真实Linear输入与可flatten stride", "CPU模块观测支持", "实际CUDA GEMM/kernel未profile", "允许称逻辑几何"),
        ("candidate为独立Q16", "mapping/history/private代码与shape", "CFG2×7、无跨候选attention", "非闭环质量证明", "允许接口事实"),
        ("B0 GPU与R0逐位一致", "六GPU输出+有效状态/RNG", "NOT_RUN", "新授权/执行", "不允许"),
        ("B0完整block性能接近原版", "同cache A/A和三轮配对", "NOT_RUN；旧R1波动大", "GPU数值/可比较计时", "不允许"),
        ("静态投影缓存可复用", "完整原形状、不可变条件、独立数值门", "两个fixture CPU控制通过", "GPU/生产失效协议", "仅静态CPU诊断"),
        ("MegaTransformer调度有增量", "R_strong/P_seq/P_sched、真实profile", "NOT_IMPLEMENTED", "D0和调度/因果数据", "不允许"),
        ("减少target调用带来机器人加速", "ΣT_round/Σg与完整质量/E2E", "本轮是block CPU回放", "有效提交/闭环独立证据", "不允许"),
    ]
    write("CLAIM-EVIDENCE-MATRIX.md", "# 主张与证据\n\n| 主张 | 所需证据 | 当前证据 | 缺口 | 允许写法 |\n|---|---|---|---|---|\n" +
          "\n".join("| " + " | ".join(row) + " |" for row in claim_rows) +
          "\n\n四项quality/TEST/非劣/正式性能权限仍false；不继承其他模型/权重或旧版本的证书。")
    source_rows = []
    old_sources = {entry["path"]: entry for entry in provenance["source_records"]}
    model_symbols = old_sources["投机采样/wam-p0-worktree/wan_va/modules/model.py"]["symbols"]
    integration_symbols = old_sources["投机采样/experiments/operator_torch_alignment_v2/integration.py"]["symbols"]
    cpp_path = root / "投机采样/experiments/operator_torch_alignment_v2/csrc/standalone_flash_sdpa.cu"
    cpp_line = next(index for index, line in enumerate(cpp_path.read_text().splitlines(), 1)
                    if line.startswith("at::Tensor sdpa("))
    for entry in provenance["source_records"]:
        if "sha256" not in entry:
            continue
        source_rows.append(f"| `{entry['path']}` | `{entry['sha256']}` | {entry['bytes']} |")
    write("SOURCE-EVIDENCE-PACK.md", f"""# 源证据包

科研base `{REVIEW_COMMIT}`；本轮派生源码的patch SHA256为 `{implementation['patch_sha256']}`。实际源文件、全部函数行号与完整SHA见[IMPLEMENTATION-MANIFEST.json](IMPLEMENTATION-MANIFEST.json)；原函数行号见[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)。只对小源/报告计算来源指纹，不对模型/tensor执行hash。

关键链：model.py WanTransformerBlock.forward:{model_symbols['WanTransformerBlock.forward']['line']}→WanAttention.forward:{model_symbols['WanAttention.forward']['line']}→custom_sdpa:{model_symbols['custom_sdpa']['line']}→Torch SDPA；既有integration.sdpa:{integration_symbols['sdpa']['line']}→standalone_flash_sdpa.cu sdpa:{cpp_line}→launch_standard/split/combine。integration.install只改60个attn_op；所有norm/linear/FFN仍由原模块计算。真实server在264–275行覆盖config的attention mode。

新B0的stage与原类分开；reference.py从固定Git对象选原AST节点，不执行整模型的import/compile，不用main替换。新的shared_history adapter只在候选history片段机械替换，原slot函数不改；native导入和CUDA执行本轮未发生。

权重：原Long checkpoint、block0、index里27个完整key/shard、shape/dtype/stride及CPU参数引用见BLOCK-PARAMETERS.json；不以mtime选权重，也不补造未计算的模型文件hash。六fixture原path/identity/bytes/encoding/layout缺口见FIXTURE-MANIFEST.json。GitHub不含原PT/权重/相机/二进制。

最终CPU证据：`attempts/cpu-real-block-final/`与`attempts/cpu-stage-layout-final/`，启动前源文本和hash已保存。早期记录以各attempt实际字段为准：最初两轮unit未保存启动源码hash；首轮手工unit也没有独立raw日志/PID出生，只保留会话工具输出和失败摘要，明确缺证、不补造。后续有登记的hash、日志、PID出生/退出方法见REGISTRATION、OWNED-PROCESS、RECEIPT，不追溯修改旧失败。

| 固定提交文件 | SHA256 | 字节 |
|---|---|---:|
""" + "\n".join(source_rows))
    write("BUDGET-AND-NEXT-STAGE.md", """# 下一阶段建议（未授权）

本轮用户附件明确ALLOW_NEW_GPU_JOBS=false；旧M22原1200秒只剩约105秒，不能借来启动新作业。当前只完成CPU授权范围。

唯一执行主线：已准备受新授权/源hash/六fixture/预算约束的GPU矩阵driver与专用worker；它还没有GPU执行证书。先在新单A100窗口做每个fixture GPU G0→G1/G2→G3与A/A，通过该fixture门后才量完整block，任何未过即停，保留已采结果且整套为PARTIAL。无新episode/训练/迁移/生产patch。

建议新GPU生命周期上限120分钟、单worker/CPU≤2、新增≤1GiB、纠错最多2次；不是自动获批。启动/环境15min、六fixture数值/状态25min、A/A与配对计时/有限profile35min、条件满足的最小冻结回归30min、清理15min。若前置门失败或缺原layout/适配即收缩覆盖、明确PARTIAL，不启用后续、不突破上限。正式性能只能无争用同边界；质量四false不变。

完整模型/完整policy回归须由其原输入与恢复契约单独绑定，不能在CPU block阶段预填PASS。只有强baseline与B0完备后，按真实profile选择一个device原语；scheduler预算不包含在此上限。

默认关键路径：10/19–20完整block GPU基线与噪声；10/21补第一个有证据的D0可行性；10/22质量/系统go-no-go；10/23冻结候选/API/claim；10/28主表；11/01 18:00北京中文证据包。乐观：GPU当日就绪，1–2天明确单block方向；常规：加1天纠错/资源排队；不利：A/A或严格数值未收口，停止scheduler实现并缩小claim。不是投稿/录用保证。
""")
    attempt_rows, subtotal, live = [], 0.0, []
    for attempt in sorted((report / "attempts").iterdir()):
        receipt_path = attempt / "RECEIPT.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            subtotal += receipt.get("lifecycle_wall_seconds", 0.)
            attempt_rows.append(dict(attempt=attempt.name, **receipt))
            if not receipt.get("owned_process_exited", False):
                live.append(attempt.name)
        elif (attempt / "RESULT.json").exists():
            attempt_rows.append(dict(attempt=attempt.name, **json.loads((attempt / "RESULT.json").read_text())))
    storage = sum(path.stat().st_size for parent in (report, code) for path in parent.rglob("*") if path.is_file())
    resource = dict(recorded_at=now, GPU_lifecycle_seconds=0, CUDA_active_seconds=0,
                    recorded_CPU_worker_lifecycle_subtotal_seconds=subtotal,
                    total_CPU_lifecycle="UNKNOWN_EXACT: first manual attempt and short interactive reads not fully timed",
                    CPU_threads_max=2, CPU_worker_wall_cap_seconds=28800,
                    current_new_storage_bytes=storage, new_storage_cap_bytes=2**30,
                    unclosed_registered_CPU_attempts=live, attempts=attempt_rows,
                    no_simulator=True, no_training=True, no_new_robot_episodes=True,
                    old_M22_budget_unchanged=True, old_remote_M17_chain_untouched=True,
                    weight_model_tensor_hash=False, **FLAGS)
    write_json("RESOURCE-AND-STOP-REPORT.json", resource)
    failures = [row for row in attempt_rows if row.get("terminal_status", "").startswith(("FAILED", "TIMEOUT"))]
    write("RESOURCE-AND-STOP-REPORT.md", f"""# 资源与止损

时间：{now}。本轮GPU生命周期0、CUDA context0、CUDA active0、模型GPU加载0、新episode/训练0。仅原checkpoint一个block加载CPU。原远端M17链继续，不把它的已授权GPU归到M24。

有收据的CPU worker生命周期累计{subtotal:.3f}秒，CPU affinity4/5、最多两核；精确总CPU生命周期UNKNOWN（首轮手工unit及交互短读未完整计时），不把可核验小计伪装全量。各attempt自然退出/真实PID出生/超时策略见JSON；原首轮PID出生NOT_CAPTURED，不补造。当前登记未闭合attempt：{live or '无'}。

本轮新增源码/报告/快照约{storage / 2**20:.3f}MiB，低于1GiB；无新模型/tensor/相机副本。报告继续写入会带来少量文本变化，最终HW落点可再核验。R00源码只读查找时误猜的patch.py不存在，已从真实索引找到guarded_worker.py；不捏造缺文件。

保留失败：{len(failures)}个已登记失败attempt（含原首轮手工unit结果）。原因包括字段归属检查错误、B14只读override错误、报告器quantile大张量限制、负例构造在inference tensor外inplace。均属于本轮CPU适配/报告器/测试问题，不能改称TF质量失败或抹掉。当前数学与状态测试不以放宽阈值取得通过。保留各轮现有原件；最初两轮unit缺启动源码hash，首轮还缺独立raw日志/PID出生，只有会话工具输出与失败摘要，明确UNKNOWN，不补造。

停止在CPU准备：没有新GPU授权，不执行GPU数值/计时/profile，不绑定生产、不开发persistent scheduler、不启动新训练/全量机器人测试。没有借M22余量、重置旧FAILED/retention/预算、重试原远端失败或清理他人进程。
""")
    write("RESULT.md", f"""# 完整block顺序基线：CPU阶段结果

生成：{now}。固定base `{REVIEW_COMMIT}`；状态 **CPU_BASELINE_ESTABLISHED_GPU_PENDING**。

A. 以前只有SDPA独立替换、原block混合执行；现在新增可单独调用、显式阶段的B0_HYBRID_SEQUENTIAL。它仍依赖Torch GEMM/norm/激活，不是全自有CUDA。

B. 六个真实action block0输入与六个同输入合法写控制，36次CPU完整执行通过R0/R0、R0/B0逻辑字节、有效KV有限及metadata/输入/RNG检查；两真实fixture各28个观察阶段、静态投影控制通过。合计仍是六个唯一真实fixture，不是独立机器人样本。

C. 原保存过程已连续化，原CUDA stride/alias/RNG缺失保持UNKNOWN。CPU与历史GPU gold不逐位，完整差异保留；B0 GPU/block以外范围均待验。四项机器人质量/正式性能权限保持false。

D. 新B0未测GPU性能。旧R1的54个原版block三轮中45点波动>5%、最大7.387倍，不能据少轮中位数保证稳定对齐。新计时先A/A再ABBA/BAAB；54行性能表都是待运行，不预填0ms/PASS。

E. 下一步先补六fixture的GPU数值/状态与可比较完整block时间，随后才依据profile选一个D0；当前不开发tile scheduler。预算与权限见[BUDGET-AND-NEXT-STAGE](BUDGET-AND-NEXT-STAGE.md)。

证据导航：[事实/未知](FACTS-AND-UNKNOWNS.md)、[覆盖](BASELINE-COVERAGE.md)、[契约/DAG](BLOCK-CONTRACT.md)、[数值字段](ACCURACY-RESULT.json)、[性能限制](PERFORMANCE-LIMITATIONS.md)、[readiness](SCHEDULER-READINESS.md)、[claim](CLAIM-EVIDENCE-MATRIX.md)、[资源](RESOURCE-AND-STOP-REPORT.md)、[源码来源](SOURCE-EVIDENCE-PACK.md)。
""")
    print(json.dumps(dict(status="CPU_REPORT_PACKET_WRITTEN_GPU_PENDING", at=now,
                          unique_fixtures=6, CPU_cases=len(cpu["rows"]), stage_fixtures=len(stage["rows"]),
                          patch_sha256=implementation["patch_sha256"], recorded_worker_seconds=subtotal), ensure_ascii=False))


if __name__ == "__main__":
    main()
