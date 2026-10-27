# RoboTwin与LIBERO的接口区别

|项目|RoboTwin固定Base|
|---|---|
|checkpoint/stats|robotwin_uncond_3cam_384.pt / 同名dataset_stats，8eaceeb|
|动作/proprio|32×14动作、1×14关节状态；官方zscore；qpos执行，无LIBERO夹爪符号变换|
|相机|head320×256，上方；left/right各160×128在下方，合成H384×W320；不做LIBERO相机翻转|
|噪声/调度|每调用seed42、CPU随机生成；10次BF16 scheduler更新、sigma_shift5|
|执行|预测32，每次最多执行24条后重规划；环境步、qpos命令、policy chunk分列|
|基线compile|官方Robo wrapper省略该参数，infer_action默认False；首轮canonical保留eager|
|环境|FastWAM固定vendor源码＋固定bf44be5 task_config；Sapien3.0.0b1、mplib.2.1、toppra.6.10、numpy1.26.4|
|任务协议|clean/randomized分开，unseen文本；expert筛有效初态，原任务horizon，失败/超时/初态异常全部保留|
|代码复用|官方factory/图像/proprio/action方法，低内存T5构造；模型与环境server-client隔离；双环境worker|
|待适配|14维方向预测器、新FIT/CAL与tau、七行候选验证与提交、三臂完整质量；没有使用7维模型充数|

CPU原始检查见远端results/CPU-CONTRACT.json；源码：官方experiments/robotwin/fastwam_policy/deploy_policy.py、configs/sim_robotwin.yaml及data/robotwin.yaml；场景与终态沿third_party/RoboTwin/script/eval_policy.py。官方配置来源：[固定clean](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/bf44be51cf5717a5595ce59447f2cf5263d2aa95/task_config/demo_clean.yml)、[固定50任务上限](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/bf44be51cf5717a5595ce59447f2cf5263d2aa95/task_config/_eval_step_limit.yml)。

语言RNG边界：固定vendor BaseTask只重设NumPy/Torch，未重设Python random；指令生成使用Python random。首轮保留官方行为，记录每个候选seed之前的Python RNG状态、生成调用N=1和实际unseen指令，不暗加重播seed。pilot每task只1有效episode，不能称原100局语言/成功率复现；后续配对须冻结同一场景/RNG与原始任务文本。
