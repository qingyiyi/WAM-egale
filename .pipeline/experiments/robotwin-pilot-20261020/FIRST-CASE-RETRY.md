# 首个完整Target case的入口修复与授权补跑

更新时间：2026-10-20T16:20:02+08:00。

## OBSERVED

- 固定六任务各19个专家可解初态已完整，114/114；角色与case顺序没有改变。
- `target-first6-2/place_object_basket-valid00`通过物理与初始观察逐位核对，但在首次模型请求前退出，0teacher、0policy、0环境命令。原因是wrapper仅关闭`eval_video_log`，原入口仍传入`eval_video_save_dir`，原程序因此尝试调用不存在的ffmpeg。
- 这是实施失败，不是算法失败。旧PARTIAL、traceback与70.743秒GPU生命周期保留，owned进程已清理。原脚本已另存旧namespace的`source-at-registration`，不把修正版伪装成旧版。
- 修正只在独立client wrapper移除该可选导出目录；原观察、相机、渲染、动作循环与成功条件不变，不安装系统ffmpeg。
- 用户明确回复“允许补跑，保留原 PARTIAL”；[机器可读授权](RETRY-FIRSTCASE-AUTHORIZATION.json)仅覆盖此局，不是普遍自动重试。
- `target-first6-3`已启动GPU1。16:20:02首局正在完整运行，3个policy chunk、48条已执行命令，物理初态和观察逐位核对通过。不是任务终态或成功率证据。
- 已闭合GPU账本2266.544秒，当前运行另计；原24h、首六局17:46:42北京deadline和准备18:20:28北京截止均未重置。
- Target已构建108条完整512-token BF16 text bank，GPU编码→CPU保存→GPU返回张量逐位相同。完整任务cached/uncached等价仍未实测，不继承LIBERO证书。

## UNKNOWN / 后继

当前case是否成功、其余五局是否能在原预算内完成尚未知。每次开局须剩余1800秒完整任务和900秒清理；不足就不启动下一局。无自动重试、无放宽horizon或删除失败。现有自动链只覆盖首六条Target，后续投机采集、训练、DEV未完整实现。

快照：[16:20原始状态](resume-evidence/snapshots/20261020T162002+0800/target-first6-3--STATUS.json)、[客户端状态](resume-evidence/snapshots/20261020T162002+0800/target-first6-3--place_object_basket-valid00--STATUS.json)、[旧失败](resume-evidence/snapshots/20261020T162002+0800/target-first6-2--place_object_basket-valid00--STATUS.json)。四项科学权限仍false。

## 后继完整终态（16:24:50核验）

获准补跑的place_object_basket/valid00已自然成功：240条实际命令、8policy chunk，client生命周期184.196563秒（含场景准备/动作/模型请求，不含常驻server此前冷启动）。失败罚分为成功实际长度240。每次policy约7.57–8.08秒；这些包含监督采集的观察计时不是正式性能结果。已自动接第二个stack_blocks_two，原六局没有全部完成。

[完整成功原件](resume-evidence/snapshots/20261020T162450+0800/target-first6-3--place_object_basket-valid00--STATUS.json)。旧视频错误仍是PARTIAL，不改为成功；本次是另一个明确获准的attempt，不能计成两个独立episode。
