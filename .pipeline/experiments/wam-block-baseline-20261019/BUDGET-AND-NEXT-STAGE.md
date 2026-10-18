# 下一阶段建议（未授权）

本轮用户附件明确ALLOW_NEW_GPU_JOBS=false；旧M22原1200秒只剩约105秒，不能借来启动新作业。当前只完成CPU授权范围。

唯一执行主线：已准备受新授权/源hash/六fixture/预算约束的GPU矩阵driver与专用worker；它还没有GPU执行证书。先在新单A100窗口做每个fixture GPU G0→G1/G2→G3与A/A，通过该fixture门后才量完整block，任何未过即停，保留已采结果且整套为PARTIAL。无新episode/训练/迁移/生产patch。

建议新GPU生命周期上限120分钟、单worker/CPU≤2、新增≤1GiB、纠错最多2次；不是自动获批。启动/环境15min、六fixture数值/状态25min、A/A与配对计时/有限profile35min、条件满足的最小冻结回归30min、清理15min。若前置门失败或缺原layout/适配即收缩覆盖、明确PARTIAL，不启用后续、不突破上限。正式性能只能无争用同边界；质量四false不变。

完整模型/完整policy回归须由其原输入与恢复契约单独绑定，不能在CPU block阶段预填PASS。只有强baseline与B0完备后，按真实profile选择一个device原语；scheduler预算不包含在此上限。

默认关键路径：10/19–20完整block GPU基线与噪声；10/21补第一个有证据的D0可行性；10/22质量/系统go-no-go；10/23冻结候选/API/claim；10/28主表；11/01 18:00北京中文证据包。乐观：GPU当日就绪，1–2天明确单block方向；常规：加1天纠错/资源排队；不利：A/A或严格数值未收口，停止scheduler实现并缩小claim。不是投稿/录用保证。
