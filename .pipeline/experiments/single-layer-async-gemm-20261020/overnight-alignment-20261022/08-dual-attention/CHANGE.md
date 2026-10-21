B14 Self/Cross attention拆为CTA内两个独立4-warp任务组，各有同样FA2 tile/workspace/参数及named barrier3/4。原四MMA warp之外的warp在attention阶段参与另一任务。2×FA2工作区大小须用编译资源核对；原GEMM/Norm/逐位算术不动，不是跨阶段DAG。有限九签名及最终尾块/CFG变体门约束。
