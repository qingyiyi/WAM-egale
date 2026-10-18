# 源证据包

科研base `ca5f37e0482c2e38e3738e3d8e71e3b185990a23`；本轮派生源码的patch SHA256为 `cb636c5bb9ef83d46b3dd880f2bc69c7e3fffe4cb84681b1db01d23e7137c533`。实际源文件、全部函数行号与完整SHA见[IMPLEMENTATION-MANIFEST.json](IMPLEMENTATION-MANIFEST.json)；原函数行号见[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)。只对小源/报告计算来源指纹，不对模型/tensor执行hash。

关键链：model.py WanTransformerBlock.forward:741→WanAttention.forward:595→custom_sdpa:37→Torch SDPA；既有integration.sdpa:14→standalone_flash_sdpa.cu sdpa:121→launch_standard/split/combine。integration.install只改60个attn_op；所有norm/linear/FFN仍由原模块计算。真实server在264–275行覆盖config的attention mode。

新B0的stage与原类分开；reference.py从固定Git对象选原AST节点，不执行整模型的import/compile，不用main替换。新的shared_history adapter只在候选history片段机械替换，原slot函数不改；native导入和CUDA执行本轮未发生。

权重：原Long checkpoint、block0、index里27个完整key/shard、shape/dtype/stride及CPU参数引用见BLOCK-PARAMETERS.json；不以mtime选权重，也不补造未计算的模型文件hash。六fixture原path/identity/bytes/encoding/layout缺口见FIXTURE-MANIFEST.json。GitHub不含原PT/权重/相机/二进制。

最终CPU证据：`attempts/cpu-real-block-final/`与`attempts/cpu-stage-layout-final/`，启动前源文本和hash已保存。早期记录以各attempt实际字段为准：最初两轮unit未保存启动源码hash；首轮手工unit也没有独立raw日志/PID出生，只保留会话工具输出和失败摘要，明确缺证、不补造。后续有登记的hash、日志、PID出生/退出方法见REGISTRATION、OWNED-PROCESS、RECEIPT，不追溯修改旧失败。

| 固定提交文件 | SHA256 | 字节 |
|---|---|---:|
| `投机采样/wam-p0-worktree/wan_va/modules/model.py` | `f579ee02b490dbffcf78add6452742dbdbef43286600c0eeae3dede3f053cb5b` | 50603 |
| `投机采样/wam-p0-worktree/wan_va/wan_va_server.py` | `bad90775a446aa1cc8d175a19ce9d43d9b7dc523c260f1bf586eee95a6e1cea1` | 155918 |
| `投机采样/experiments/operator_torch_alignment_v2/integration.py` | `ff4b7dd8ca820b563725a4719de2117d12fe4e244eaa733fdcc36201cab96f36` | 1272 |
| `投机采样/experiments/operator_torch_alignment_v2/setup.py` | `436a1605e63c8f775b63d0ac73708be10e528f866a77a4084236c1f8db8e3598` | 1043 |
| `投机采样/experiments/operator_torch_alignment_v2/csrc/standalone_flash_sdpa.cu` | `44441db658b0a8d4fab693fb757fc09d465ba6fb78c1f098cc3be0f046f5dde1` | 9610 |
| `投机采样/experiments/operator_torch_alignment_v2/server_entry.py` | `e96c5f9a3f1ee660388587aa5e55492e82a6c7cc82bc5bec9acd976f962fc094` | 1096 |
| `投机采样/experiments/operator_torch_alignment_v2/worker_full.py` | `bd848e7a165b1fb84835132172d10db2389341170780dbd9dd9a2b847af0ef4a` | 29122 |
| `投机采样/experiments/operator_torch_alignment_v2/worker_performance.py` | `06c682826fbc68b332cc22c7b862efb11568be8d229223547e9c76fbf7442256` | 23361 |
| `投机采样/experiments/operator_torch_alignment_v2/worker_full30_performance.py` | `5fe0eb65530fc82e5cff9862013fa59d3eb5aa4c96f26a4e0719f00068c67fa0` | 5865 |
| `投机采样/experiments/operator_torch_alignment_v2/saved_input_validate.py` | `21d916a914accd8bd35ee2200d0fdba28b851eadaf31ca914113fa72539da938` | 6081 |
| `投机采样/experiments/shared_prefix_torch_v2/csrc/shared_torch.cu` | `1ed851fd57652136591a97d043b913b2e2edcf51121eb404c8624990a9d1592d` | 6834 |
| `投机采样/experiments/shared_prefix_torch_v2/csrc/shared_flash_fwd_kernel.h` | `7c7c7ec26149613e591022b41045da715d21d99bdd493ec6ef2e2171b700625a` | 24108 |
| `投机采样/experiments/shared_prefix_torch_v2/csrc/shared_load.h` | `4c0452f576b38823b85b553f281a32086df609c8c8b1b70bd4b2905bca89a394` | 2528 |
| `投机采样/experiments/wam_candidate_tile_v1/csrc/group.cu` | `41ebcde7958e8358dbce3bcdd83d84abf53511fc1cfd0ba7e1ba88ff32a5726d` | 4880 |
| `投机采样/experiments/wam_candidate_tile_v1/csrc/group_kernel.h` | `0b3fab70086f033bb9eec326d66be2347d50cafeb4a0cb33462d88ca4002627d` | 24403 |
| `投机采样/experiments/wam_candidate_tile_v1/csrc/group_load.h` | `5c9c55c6fa02405c30eed5fe19392613ee73f310a1e93056e9b4b1e07b1d4f84` | 2577 |
| `投机采样/experiments/wam_candidate_tile_guarded_v2/guarded_worker.py` | `577327e4a676d8286d2fa1f4af4fd91421d42747f74a72b672fcf9b8624fe47d` | 14216 |
| `投机采样/experiments/wam_static_prompt_baseline_v2/static_projection.py` | `ee3c03fd6c125420da075a5746d9cc2d94454372d71b4422ca41449fa599c0d2` | 5833 |
| `投机采样/experiments/wam_static_prompt_baseline_v2/baseline_contract.py` | `acddf3da321cd1f2bb947018e3dd544cd8988bfe1b92819b97ca2429b896c6fe` | 10546 |
| `投机采样/experiments/single_worker_chc_gpu_v4/adapter.py` | `ca4cda2ee776dde6ea82471eba2b192010eb0787032fafc617b466320388a4e7` | 23211 |
| `投机采样/wam-p0-worktree/LICENSE.txt` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | 11357 |
| `review/CURRENT-PROGRESS-20261018.md` | `272c86a89ec96e740c3a565a0968694596638487b91066750cbb483616a5de8c` | 959 |
| `review/PROJECT-REVIEW-PROMPT-20261018.md` | `d11a955e6e6306516b9ec724663e3cce3a0e29af3ac909d739004f9ac4dd6ae3` | 5825 |
| `.pipeline/cycles/wam-research-20261016/STATUS-20261018.md` | `e053e76f97df0dfd3d18b1c9159598436f191e3cee5869eecb47a7e6ed116f2d` | 6608 |
| `.pipeline/experiments/operator-torch-alignment-20261018/RESULT.json` | `83b1970284a8b559ab41790b1354137624da062336a706082326dbf191cd9a7d` | 2646 |
| `.pipeline/experiments/operator-torch-alignment-20261018/RESULT.md` | `dca68605990a1025cc58866bf062ca3ce36021e242ea8d2d29f57664ac25f254` | 4690 |
| `.pipeline/experiments/operator-torch-alignment-20261018/ATTENTION.csv` | `5c00c108368061989d00a0cbe7cc0e9761c985855da81d64ad75052eaa9cd40d` | 4639 |
| `.pipeline/experiments/operator-torch-alignment-20261018/BLOCK.csv` | `b7fd14adfc213fabe7835639cb2279fc20b7ce761aaddcd4826ad96df8253297` | 2615 |
| `.pipeline/experiments/operator-torch-alignment-20261018/FULL30.csv` | `e710cbef6608fd5e5b381ac0a851357e0bfe0ecf45d6c2d91d774cf64fb6e614` | 1469 |
| `.pipeline/experiments/operator-torch-alignment-20261018/POLICY.csv` | `c25e51e53454e42def4de03d861a98bd993c2777eaf9868af369dd59acb6d2cb` | 927 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/RESULT.json` | `bf994fae3d8f0a25cae8f6abbab7849ffd8a1c8386ea2256c1d88e7655245444` | 4741 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/RESULT.md` | `2d4521e02181117f8057b6ccafb4c2003e4bfb87dfa194e574e1541453509adb` | 3531 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/PERFORMANCE-ONLY.csv` | `7bb8073ecd5391bc2d3810ff922768b4b03674f4584d833c70e65f261f2a7ab7` | 232569 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/PERFORMANCE-ONLY-ROUNDS.csv` | `79652e7d5981d21738de372479c1a9c1daa5a91ce7e8a3dd5cde8ed0fc4143ed` | 417318 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/FULL30.csv` | `c6732ea1f68603c30e75eb08da7b8fb55652d6e800cfabed41a941427a3a9570` | 806 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/FULL30-ROUNDS.csv` | `0ba729dcf42530e417a340478875e777ac63bcbfce1b3ccea10f1a2ec3bdc441` | 2027 |
| `.pipeline/experiments/operator-torch-alignment-20261018/performance/CPU-INPUT-IDENTITY-AUDIT.json` | `42bde694f6aaf28ced172ba3cdbdee1a1c69a02cba1a40c77655837aff25ac78` | 2782 |
