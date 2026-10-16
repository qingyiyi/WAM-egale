# 历史 Long 数据数值审核

2026-10-15 16:38 北京时间完成，状态 `PASS_COMPLETE_NUMERIC_AUDIT`。隔离脚本只读历史数据，CPU14/15、最多2线程、nice10、CUDA不可见；没有模型前向、环境运行、联网或旧产物写入。实际审核耗时246.9秒。

## 实测结果

| 用途 | episode | chunk | 可用相邻步监督行 |
|---|---:|---:|---:|
| FIT，init0–17，排除task0/1的init6–8 | 174 | 3259 | 159691 |
| CAL20，init18/19 | 20 | 413 | 20237 |
| CAL50，init20–24，仅数值审核 | 50 | 1005 | 49245 |

历史TRAIN194的3672个chunk确实保存了50步实际 `x_i / d_cfg / x_next`、原始Transformer输出以及原RGB和prompt，**不是只有动作终点或元数据**。每chunk导出step1–49，共179928行；前一方向严格来自同chunk上一步 `d_cfg`，状态和标签取同一步实际 `x_i / d_cfg`。

4677份TRAIN194+CAL50 trace全部通过任务/初态/chunk/frame绑定，BF16 `[1,30,4,4,1]` 状态和方向、BF16 `[2,16,30]` 原始输出、RGB、视频latent、实际σ/timestep和有限性检查。CFG1的原始条件输出还原后与 `d_cfg` 逐位相等；带首帧prefix恢复的BF16 Euler重建最大误差0；有效7通道的相邻状态和最终动作均逐位连续。`max |timestep−1000×sigma|=2.002716064453125e−5`。

全部229173个相邻步在完整30通道上有差别，最大0.181640625，差别仅在会被下一次输入准备清零的未使用通道；112维有效状态全部一致。4677个最终动作文件也经过未使用通道清零，不能将这些正常准备操作误报为数据断裂。

原teacher checkpoint的ID/路径与Long合同相符；历史采集没有保存完整权重字节清单，本次不会事后补作追溯认证。每行是**历史target-only实际查询监督**，不是当前CFG14 verifier在新候选状态上的oracle，没有虚构node0/tree身份。hidden/KV未捕获不影响这些已有方向用于预训练，但不能将其直接改名为新候选标签。

## 现成same-query辅助池

原11091行Long10辅助池的3697份teacher文件全部回读，与缓存逐字段、逐tensor相等；每树3节点，状态hash与实际target查询hash一致，state/previous/target和时间标量全部有限。来源为task0–9各init17，包含失败长轨迹，不按成功筛选；生成器是旧 `B_MIXED_HISTORICAL_C` 研究checkpoint，不是当前冻结A_off。原current3906行也通过缓存字段/数值/实际state绑定检查，其来源为task0/1的init0/1；本次未再逐一回读这3906行的1302份原teacher文件。

current3906+aux11091共14997行均属于FIT，与CAL20或CAL50没有episode交叉；init17不能同时作为新CAL。它们的标签已有效，不必重采；新TF部署访问不同状态时，仍须另外检验分布偏移与完整闭环质量。

## 产物与加载

结果目录：`投机采样/results/runtime/p0-data-numeric-audit-v1-20261015-r1/`。

- `RESULT.json`：总状态与pack关联；必须为 `PASS_COMPLETE_NUMERIC_AUDIT` 才使用产物。
- `HISTORY-RESULT.json`：244条历史轨迹数值审核、逐episode用途和实际统计。
- `SAME-QUERY-RESULT.json`：11091辅助行的3697个原来源、生成器/teacher/计划绑定和逐任务节点数量。
- `historical_train194_columnar.pt`：252809145字节，约241.1MiB。**CAL50数据没有导出到这个pack。**

```python
pack = torch.load(PACK, map_location='cpu', weights_only=False)
c = pack['columns']
fit = pack['views']['fit174']
cal = pack['views']['cal20']
features = torch.cat([c['state'][fit], c['previous'][fit], c['scalars'][fit]], 1)
labels = c['target'][fit]
```

`state/previous/target`为FP32 `[179928,112]`；`scalars`为FP32 `[179928,3]`，顺序固定为 `sigma,next_sigma,timestep/1000`，**不可再次缩放timestep列**；`delta`为FP32。`task/init/episode/rollout_chunk/frame/step/source_id`为逐行int32，其中 `episode=task*100+init`；`sources`保留每chunk源路径、大小、mtime、prompt、teacher ID/path和对应行区间。未拟合任何归一化；后续只用FIT174拟合统计。

## 数据复用与最小补标顺序

1. 直接用已审核FIT174做相同数据、抽样、预算和三seed的TF64/MLP96预训练，CAL20仅用于选型。保留旧DEV原有用途，不将其变成独立测试。
2. 若需要部署分布校正，预声明同样的current3906/aux11091抽样与权重，并对两结构一致使用；不将历史target-only行伪造为root，也不因为某seed更好而事后更换协议。
3. 固定候选后才补新候选访问状态的teacher。源码已有显式counterfactual接口：初始观测、终态video cache、只读candidate target查询、观测/动作cache恢复。旧RGB/latents/actions为复用这些接口提供数据；**完整context回放与查询等价尚未GPU验收**，尤其动作postprocess→preprocess及KV/VAE历史，不能提前宣布重建成功。通过等价门后可以避免重新执行物理环境来补部分标签。
4. 最终任务成功率、轨迹步数及自然耗时仍由完整LIBERO配对闭环决定。target-only预训练loss或记录状态上的接受代理不能替代这一步。

仅审核TRAIN194与CAL50的tensor；质量init25–29、未来留出和旧保留DEV的tensor storage均未读取。已有aux缓存所在archive含DEV对象元数据，但本脚本没有访问其tensor内容、评估或导出DEV。
