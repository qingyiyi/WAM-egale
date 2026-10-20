# M37设备数值契约：四项NO-GO，有界收口

OBSERVED：当前远端公共cuBLASLt描述符重建确认4个Q/FFN2的tile、stage、split-K与预先候选依据一致；不是原Torch内部调用拦截。实现使用固定CUTLASS v3.5.1（f7b19de32c5d1f3cedfc735c2849f12b537522ee）的真实SM80 multistage mainloop、epilogue与serial split-K semaphore，不调用host BLAS计算候选，不重复旧四种仅改变workspace累加公式。该组件单独编译成功，不是整层或warp职责流水实现。

|同输入原语|逻辑M×N×K|不同元素|最大绝对差|相对L2|原逐位门|
|---|---|---|---|---|---|
|B2 Q|32×3072×3072|68/98304|0.0625|0.01560%|FAIL|
|B2 FFN2|32×3072×14336|33316/98304|0.25|0.23120%|FAIL|
|B14 Q|224×3072×3072|173734/688128|0.0625|0.23769%|FAIL|
|B14 FFN2|224×3072×14336|154790/688128|0.25|0.21039%|FAIL|

四项候选重复调用自身逐位一致，输入/参数与CPU/CUDA RNG保持；四项原Torch比较全部失败。Q/FFN2金标准直接使用原Torch Linear，输入来自已冻结顺序native的同一次debug，而不是失败WMMA的上游输出。没有把隐藏激活误差当成物理动作误差。

宏几何：B2 Q逻辑tile64×64、K64/stage5/split1；B2 FFN2 tile64×256、K64/stage3/split7；B14二者tile128×128、K64/stage3/split2。候选warp shape由实现选择，不能说已读取vendor warp shape；B14 Q vendor CTA swizzle=1，候选采用CUTLASS identity swizzle。公共metadata不揭示Tensor Core accumulator映射、K归约/舍入细节和全部epilogue。不能把宏几何门当作全实现一致，更不能由此单独指定失败根因。

HYPOTHESIS：设备累加、split归约和epilogue顺序中仍存在未匹配差异。当前结果只排除了“匹配这些公共宏参数即可自动逐位”的方案，不证明任何特定算子或误差是机器人失败根因。

UNKNOWN：vendor内部算术顺序，attention重新嵌入8warp执行器后的完整数值契约，新组件的正式性能。没有反汇编闭源库、修改Torch matmul设置、降低原门或以小误差发PASS。

RECOMMENDATION：这一几何匹配路线止损，停止异步整层接入；保留已对齐baseline和上次整层未准入原型。后继若要采用新算术，需要独立登记数值及闭环重验；若保持原逐位契约，必须先有可验证的实际设备实现依据，而非继续盲猜split-K。原Torch host包装调用cuBLASLt，并不提供其内部设备代码，复制host调用链不能解决单kernel内复用问题。

预算实际：4个参考block、12原语调用（4原Torch、8候选含重复）、8公共heuristic、0full30/机器人局。总预留生命周期185.698秒（包含首次入口失败、人工修正等待、CPU编译和清理），CUDA active时间UNKNOWN；不是把全部生命周期记成CUDA时间。全部child root已真实reap，最终owned活动0、两卡0MiB/0%。[资源](RESOURCE-AND-STOP-REPORT.md)。

最初metadata-1因冻结baseline DSO中的本机绝对依赖而在第一个block前失败，0block/0heuristic。原始失败、原return0但status FAILED、清理与耗时均保留；新entry-corrected-1只替换本轮debug入口为既有远端sequential数学链，不改其源码/DSO，不重置T0或截止。[实施记录](ENTRY-IMPLEMENTATION-ISSUE.md)。

源码：[contract_linear.cu](../../../算子研究/wam_ops/optimized/csrc/contract_linear.cu)、[独立构建](../../../算子研究/wam_ops/optimized/build_contract.py)、[同输入核验](validate_contract.py)。原始结果：[4原语](remote-evidence/entry-corrected-1/contract-1/RESULT.json)、[8公共描述符](remote-evidence/entry-corrected-1/metadata-1/RESULT.json)、[原失败](remote-evidence/metadata-1/RESULT.json)、[最终资源](remote-evidence/FINAL-RESOURCE-OBSERVATION.json)。

旧FAILED/NO_GO/四false、M34/M35原终态、默认A_off不变；baseline/投机/原tile与保护文件不变。没有新性能或质量放行。
