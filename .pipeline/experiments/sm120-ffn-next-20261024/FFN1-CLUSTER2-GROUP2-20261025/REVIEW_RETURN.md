# FFN1共享A两段交接：结果

## OBSERVED

本轮是FFN1＋GELU输入交接实验，FFN2七段K2048及其BF16递推未改变。A2把相邻两个K64的8KiB A合为16KiB DSM，tile/1T4M3E/算术次序保持；实际计数确认每tile的DSM/credit/ACK从48降为24。使用原双槽、没有增加shared。

三档真实B14/Q16输入、两个独立进程完成27张量、七段partial/running、有效KV、原输入、RNG、自定义stream和真实计数核验，有限逐位通过。170CTA、100352B shared、255寄存器/线程、784B local均与旧实验相同。不能外推所有输入或闭环。

|同轮无插桩完整层|短KV288|中KV1152|长KV2160|
|---|---:|---:|---:|
|Torch/A2|0.976×|1.390×|1.770×|
|Torch/旧共享A|1.019×|1.443×|1.831×|
|Torch/保留S|1.286×|1.789×|2.229×|
|A2相对S延迟|+31.73%|+28.77%|+25.92%|
|A2相对旧A延迟|+4.35%|+3.86%|+3.44%|

同轮A2 FFN1约355μs，独立加载W约152μs；原共享A约323μs是历史阶段观测，不冒充本轮重测阶段。FFN1＋FFN2的插桩包络约489–491μs。完整层与阶段分开，阶段区间不可相加替代完整层。

## HYPOTHESIS / UNKNOWN

源码明确：为把现有两槽合成一次DSM，A2等两份A都ready才发送，下一组需两槽empty；旧方案可step+2提前装载。减少握手同时失去装载/消费重叠，观测支持这项代价超过本次收益。该实验没有单独分离每种等待贡献，不能断言差值全由DSM带宽或某条指令造成；没有NCU计数器、没有HBM流量实测。

逻辑A/B读数及DSM字节未因合并而减少；只减少交接次数。不把逻辑请求减少称为HBM节省。

## RECOMMENDATION / 采用决定

NO_GO：不采用A2，保留S及独立二进制。两个进程、三KV均退化，停止本轮，不继续盲目扩大传输块。后续更值得先对齐S的FFN1 GEMM与原Torch，再评估不阻断原流水的复用；这是建议，不是已经启动的新设计。

## 固定入口

- [新候选源码](whole_layer.cu)：`action_ffn1_produce`，两个K64一组；旧未调用helper保留在冻结快照中，不是实际新producer路径。
- [逐阶段表/快照](STAGE-TABLE.md)，[汇总](SUMMARY.json)。
- [进程1原数据](measure-1/RESULT.json)、[进程2原数据](measure-2/RESULT.json)，同一fixture同时用于精度与计时。
- [正常构建](build_candidate.py)、[插桩生成](make_profile.py)、[实际运行](run.py)。普通/插桩DSO独立，插桩计时不入整层主表。
- `S.so`、`W.so`、`A_SHARE.so`、`L.so`原二进制独立；采用源码仍为`../../rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`。
- [资源/停止](RESOURCE-AND-STOP-REPORT.json)、[清理收据](OWNED-CLEANUP.json)、[IO暂停](IO-PAUSE.json)/[恢复](IO-RESUME.json)。迁移自有进程只在计时窗口暂停，其他用户任务未操作。

四项科学权限false不变，本实验不是机器人任务性能或质量证书。
