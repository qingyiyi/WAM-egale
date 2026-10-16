# 今日同一常驻target臂的四完整重复

只读取事前登记task2/8 init33的rep0/1共4局；不读取TF/MLP/fresh/risk结果，不改今日173源或冻结规则。原horizon完整episode与全saved helper通过，不能把这称为整个T臂或50局终态通过。

| task | rep0/rep1成功 | 实际步数 | 初物理/RESET/RNG | 初图 | 全commands |
|---|---|---|---|---|---|
| 2 | 均成功 | 258 / 263 | 保存范围相同 | 不同 | 不同 |
| 8 | 均成功 | 536 / 382 | 保存范围相同 | 不同 | 不同 |

两对初始两相机差异及首命令位置：
- task2：[{"path": "root/observation.images.eye_in_hand_rgb", "changed_pixels": 1, "changed_elements": 3, "max_abs": 1.0, "first_index": [44, 28, 0]}]；首命令不同位置（从0计）[0, 0]。
- task8：[{"path": "root/observation.images.eye_in_hand_rgb", "changed_pixels": 2, "changed_elements": 6, "max_abs": 2.0, "first_index": [14, 110, 0]}]；首命令不同位置（从0计）[0, 0]。

两对都在初始像素已有差异时，首policy完整actions_0/latents_0已不同。现有记录只证明这条时间顺序，不能认定像素导致后续动作或步数变化。不同尾部未强行对齐；完整共同policy/cache输出均已逐字节比较，详情见REPEATABILITY.json。首cache实际12个逐步观测，每执行1条非终止命令存1帧；不作每4动作一帧的错误映射。

与旧跨r3/r4结果分别报告：旧T task2初图相同、首动作/视频完全相同，执行第一条相同命令后的历史腕图已不同，第二chunk首次命令不同；旧T task8初图不同且成功393→失败796。今日两个task在同一登记常驻T臂内两次均成功，但task8长度536/382仍不同。两种观察不能混成额外独立质量样本，也不能据今日4成功把旧失败修成PASS或声称评测可保证重复。

本次实际读取93份actions、93份latents、89份obs，共275PT/3002数组，全部finite；另读4局commands、初终sim/controller保存字段、初图、metadata/RESET/COUNTER/FINISH，以及唯一T arm的实际CPU/CUDA初RNG张量。该arm-level快照已与READY指纹匹配，四个RESET记录也一致；没有四份独立的per-episode RNG原对象。

原始Python/NumPy RNG只有hash，未存对象；逐步完整模拟器内部状态、每请求噪声/KV/VAE中间原张量没有保存。不能据保存初态相同推定全部内部状态相同。

本次CPU数值审核7.36秒，GPU/SSH/forward/新episode为0；全部新结果只在独立审核目录。之后等今日50全部真实终态，再按已冻结协议完整复核；不利用这次结果改变原选择条件、seed或阈值。
