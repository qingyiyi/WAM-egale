# 迁移状态（2026-10-25T21:11:21.264966+08:00）

RoboTwin已在5090进入完整DEV，使用原full512 BF16 reset bank与原模拟器/渲染配置，单worker。首个place_object_basket/valid49 Target完整失败，原任务step_limit700，尚无完整四臂组；OLD正在接续。此新Target与M41同名case旧Target成功不同，保留差异，后续核初态/输入/版本；不借旧成功替换本次结果，也不称已确定阈值。

原始server01模型准备因AST行号处理漏项退出，0teacher/episode；复用M41已验证annotate修复后独立server02/panel02，原失败不改。实际server/client导入Torch2.9/cu130、numpy1.26.4，CuRobo五模块SM120构建完成。没有smoke、模型hash、改驱动或新训练。

本次实际server峰值allocated约20.70GiB、reserved约22.13GiB，另有模拟器显存，因此32GiB上不盲开双worker。原21:56:53硬截止及整局1800秒＋清理600秒预留不变，可能只能形成PARTIAL面板。

本机watch_finish每60秒巡检已启动队列，无重试或新GPU任务；终态后owned身份核查、rsync回传与fsync/清单/关键JSON核验完成才删除本次临时5090根，保留算子工作区。原四科学门false。实际状态继续读取WATCH-STATUS/LIVE-SNAPSHOT；本页为快照。
