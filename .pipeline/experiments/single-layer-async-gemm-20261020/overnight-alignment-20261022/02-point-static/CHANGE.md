pointwise阶段静态CTA-stride分配同一2048元素tile，移除队列atomic/编号广播及无shared数据的tile结束屏障；起止grid.sync不变。所有逐元素公式、FP32/BF16/FP64边界、RMS/GEMM/attention保持。B2/视频亦通过原九签名门后才选择。基于原043c7759，不叠加无稳定收益的Self-Q路由控制。
