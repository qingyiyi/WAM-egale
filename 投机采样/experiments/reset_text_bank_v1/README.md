# Long10 CPU text bank

Build all ten public LIBERO-Long task embeddings plus the empty CFG negative
prompt with the existing source's exact text helper and target T5/tokenizer.
This is a reusable 44 MiB inference-input artifact and text-stage measurement,
not a smoke rollout or a GPU speed benchmark. No running server is modified.

The builder loads only the CPU text encoder. It requires CUDA invisible,
records tokenizer and T5-forward timing separately, verifies finite BF16
`[1,512,4096]` tensors and unchanged inference RNG, and verifies the complete
saved bank exactly on reload. It uses at most eight CPU threads and a finite
30-minute deadline. Existing job paths are refused.

Before deployment, an isolated server still needs to validate the source and
weight identity, CPU/GPU output transfer, complete reset state equivalence and
full-task trajectories. Cached outputs must be applied to both T and the
speculative candidate. The current AutoDL plan remains unchanged. This bank
also provides frozen text features for a future conditioned draft; it does
not train or select that architecture.

`bank.py` is an isolated, default-off adapter. It checks the bank's runtime,
source and target identities, only serves covered single-prompt BF16/512-token
requests, returns independent storage, and rejects unsupported requests.
It changes only `encode_prompt`; KV/VAE/controller/RNG reset remains in the
original code. It has not been installed in the AutoDL or production server.

## 2026-10-15 实际结果

11/11个真实文本输出已完成，约44MiB；CPU编码706.01秒中T5前向705.81秒。
保存重载和缓存CPU查找逐位一致、返回值修改不污染bank、错误请求拒绝，RNG不变。
这不是整个server reset或GPU/E2E验收；所有在线部署仍默认关闭。

原始pad512→只编码有效token的完整11输入对照也已执行：69.38秒，但0/11逐位一致，
最大绝对差0.16015625，保持失败并不接入冻结target。

- [CPU bank结果](../../results/runtime/p0-reset-text-bank-v1-20261015-cpu7-r2/RESULT.json)
- [缓存CPU检查](../../results/runtime/p0-reset-text-bank-v1-20261015-cpu7-r2/CACHE-CPU-CHECK.json)
- [裁padding负结果](../../results/runtime/p0-reset-trimmed-text-v1-20261015-cpu7/RESULT.json)
