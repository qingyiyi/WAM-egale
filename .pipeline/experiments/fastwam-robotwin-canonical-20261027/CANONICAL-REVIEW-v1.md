# RoboTwin首四任务canonical审阅

固定Base14维、三相机、32预测/24执行、十步shift5；官方Robo wrapper默认eager。两隔离客户端共享一模型，四局全部完整成功；这是工程pilot，不是论文成功率复现。

|任务|实际scene seed|成功|动作命令|policy chunk|并行环境wall秒|
|---|---:|---|---:|---:|---:|
|adjust_bottle|4300001|True|118|5|69.29|
|beat_block_hammer|4300000|True|105|5|59.34|
|blocks_ranking_rgb|4300000|True|436|19|100.20|
|blocks_ranking_size|4300000|True|445|19|100.49|

全部真实执行命令等于每次输出前24条的合法前缀；每chunk G=10，scheduler网格一致、strict模型加载通过。四份三相机contact sheet已实际审阅；最后图片为动作执行前，不单独证明终态，成功采用官方环境记录。

adjust_bottle的初始候选4300000存在一次expert_setup_exception_logged，保留后按官方筛选到4300001；其余三项均4300000。前三attempt因GL/EGL/录像入口失败，0policy调用，全部原记录保留。修复仅使用用户目录既有库/ffmpeg路径，无系统驱动/服务变更。

成功attempt04 GPU生命周期232.508秒；加前三失败与两次诊断总计502.120秒，原5400秒canonical预算未重置。该并行wall时间仅作资源估算，不是加速比。

技术与图像工程门通过；四科学权限仍false。下一步登记Robo自身完整episode隔离的FIT/CAL/DEV与三卡分片，先验证真实14D采集/批验证数值，再训练和重新校准阈值。未训练Robo预测器，未取得Robo三臂成功率或5090速度。正式全量为50任务×100有效初态×clean/randomized×3臂=30000局；本轮4局不能替代。
