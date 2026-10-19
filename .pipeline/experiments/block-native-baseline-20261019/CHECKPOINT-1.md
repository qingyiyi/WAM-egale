# 完整native首轮门

六个真实完整block0，六组全部逐位通过；每组23个已匹配关键中间量/输出，另6个同query原语隔离门。有效KV/全部metadata、输入、RNG、27份参数不变。FFN2复现采用固定Torch cuBLASLt descriptor/heuristic/workspace路径，属于复用成熟GEMM而非自写MMA。

共享GPU仅精度，不记录性能。未验证驱逐、write1/2或完整30层，不切默认入口。旧CUTLASS FFN2失败不改。原始结果：[native-gates-attempt1/data/RESULT.json](native-gates-attempt1/data/RESULT.json)。
