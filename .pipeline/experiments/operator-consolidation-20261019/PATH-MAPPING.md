# 算子路径迁移表

原16目录已删除。历史源码提交：`a0ed2d8e49403cdca8d87afd40b21dd19595710c`。所有旧实验结果目录保留；下表不修改旧FAILED或身份。

| 旧目录 | 当前去向 |
|---|---|
| block_sequential_baseline_v1 | 当前baseline与共享工具；旧tap/多引擎/多轮runner退役 |
| direct_torch_block_copy_v1 | 算子研究/wam_ops/baseline；共享install与tools |
| operator_kv_matrix_v1 | 共享tools/replay.py；旧宽矩阵代码退役 |
| operator_torch_alignment_v2 | 算子研究/wam_ops/optimized（同CUDA数学源、重新构建/命名） |
| wam_candidate_tile_chunk_correctness_budget_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_chunk_correctness_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_chunk_correctness_v2 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_chunk_result_audit_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_fixture_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_graph_pilot_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_guarded_v2 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_profile_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_candidate_tile_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_shape_audit_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_static_prompt_baseline_v1 | 仅Git历史，当前未安装；原始输入和结果仍保留 |
| wam_static_prompt_baseline_v2 | 仅Git历史，当前未安装；原始输入和结果仍保留 |

投机侧shared-prefix及其原tile构建依赖不在删除范围，原模型/配置/训练/接受器/调度未修改。
