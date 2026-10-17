# 已记录风险对的组件诊断

终态：COMPLETE_RECORDED_COMPONENT_DIAGNOSTIC。仅检查既有四局，未运行模型或新 episode。

|task/init|arm|成功|实际环境步|800失败罚分|命令/保存输出组件审计|
|---|---|---|---|---|---|
|1/29|T|True|264|264|RECORDED_EPISODE_COMPONENTS_VERIFIED|
|1/29|TF_ON|False|796|800|RECORDED_EPISODE_COMPONENTS_VERIFIED|
|3/26|T|True|240|240|RECORDED_EPISODE_COMPONENTS_VERIFIED|
|3/26|TF_ON|False|796|800|RECORDED_EPISODE_COMPONENTS_VERIFIED|

|task/init|sim/controller数组精确|CPU/CUDA RNG精确|完整reset精确|两相机像素精确|
|---|---|---|---|---|
|1/29|True|{'CPU': True, 'CUDA': True}|True|{'observation.images.agentview_rgb': True, 'observation.images.eye_in_hand_rgb': True}|
|3/26|True|{'CPU': True, 'CUDA': True}|True|{'observation.images.agentview_rgb': True, 'observation.images.eye_in_hand_rgb': True}|

原 namespace 的 FAILED/PARTIAL 和 TF_ON 局部 arm 清理收据缺口全部保留。全局退出证明单独核验，不能补造局部收据。
四项原权限全部 false；strict quality gate 仍阻塞。本报告不生成质量证书，不推断失败根因。
每相机像素差异、实际 RNG 指纹、四局命令与保存输出细节见 RESULT.json。
完整局耗时属于共享资源且带记录的描述，ΣT_round/Σg 未采集，均为 UNKNOWN。
