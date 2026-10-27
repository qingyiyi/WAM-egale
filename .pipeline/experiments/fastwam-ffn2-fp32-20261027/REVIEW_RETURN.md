# B7 FFN2连续FP32：完成，暂不采用

结论：允许评估小数值差异是合理研究方向；本次单一改动未获得完整层收益，因此不值得据此承担新的闭环验证成本。严格TMA＋1T4M3E仍为当前保留算子，投机默认Torch。

OBSERVED：仅B7 FFN2由16次BF16中间提交改为完整K4096连续FP32累加、最终bias/BF16一次。TMA双缓冲、1T4M3E、tile64×128、K64、其它阶段和RMS/LN归约不改。B1数学不变。

|B7 layer15|Torch eager|严格版|候选|
|---|---:|---:|---:|
|FFN2设备µs|43.211|66.076|62.325|
|完整层设备µs|568.345|594.293|599.004|
|直接API完整层µs，无外部Graph|919.907|638.557|643.167|

局部快约6.02%，延迟下降5.68%；整层候选反而慢.79%（小于1%，不能当稳定退化定论，但没有正收益）。官方compile完整层设备293.205µs，依旧明显快；该数值/性能基线保留，不能用eager结果声称胜compile。对称Graph仅用于拆GPU设备成本，两侧同边界；不把它伪装成无Graph测量。直接API编译对照本轮关闭内部Graph，不能冒充官方reduce-overhead部署速度。

三代表层B7 FFN2 relL2=.436–.460%，整层=.050–.342%；layer15候选误差小于compile/eager，但不是全层统一结论。局部更接近FP64数学参照。四固定DEV观测30层/10步动作relL2=.337–.414%、maxabs=.0078125；12树72边中1边翻转、1路径变化，夹爪符号未变，未执行环境。严格版在实际树上720次FFN2及所有teacher/动作字节一致。

HYPOTHESIS：省去中间舍入/提交有局部价值；完整kernel的代码生成/寄存器与小样本计时可能抵消局部收益。候选regs233对231、shared84992B及1CTA/SM相同，不能直接断言occupancy降低。无改动QKV隔离行出现较大差异，追加交错计时被别的会话占卡拦截，原因未定。

UNKNOWN：新路径是否改变机器人成功率/自然trace，是否跨任务泛化；本次没有质量通过结论。四科学权限保持false。

RECOMMENDATION：本候选保留为显式研究flag、不采用生产。下一步先在严格舍入契约下改善B7输出投影/FFN2的32个输出tile并行度及有序提交成本；若后续数值候选有明确整层收益，再前瞻登记动作/接受与闭环验收，不以compile偏差作容差通行证。

源码：当前optimized/csrc/fastwam_tma_pipeline.cuh:52–58（研究开关）；build_whole_layer.py默认FASTWAM_FFN2_FP32=0。固定本轮源码source-final/optimized，二进制binary/fastwam_sm120_ffn2_fp32_cuda.so；严格独立DSO和源码在../fastwam-tma-warp-port-20261027/binary、source-final。构建命令见remote/build-attempt01/JOB.json与BUILD-AND-FINAL-RESOURCE.txt。无整模型hash。

证据：[完整阶段表](STAGE-TABLE.md)、[数值](NUMERICAL-RESULT.md)、[实际接受](ACCEPTANCE-RESULT.md)、[预算/失败/清理](RESOURCE-AND-STOP-REPORT.md)。首轮测量脚本alias问题保留为INVALID；唯一修订改测试副本而非改数学。所有有效结果来自compare-attempt02和propagation-attempt01；timing-confirm01未执行，不构成第二轮性能证据。
