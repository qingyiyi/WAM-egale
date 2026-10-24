---
kind: requirement
name: Codex临时文件统一使用账户tmp目录
level: constraint
scope: account:cxz;Codex;temporary-files
status: active
updated: '2026-10-24'
sources:
  - 用户要求清理我们放在/tmp中的文件，并在bashrc设置Codex默认临时目录为/rjs/cxz/tmp
---
# Codex临时文件统一使用账户tmp目录

新临时文件和临时脚本默认写入/rjs/cxz/tmp。按用户要求在/rjs/cxz/.bashrc设置TMPDIR、TMP、TEMP，并在Codex子进程环境配置相同路径，以覆盖不加载bashrc的启动方式。已有进程的环境不能靠修改配置追溯改变；当前会话新命令显式使用新目录，后续新会话读取配置。

此次用户授权清理项目外/tmp中本账号已确认不在使用的临时文件，扩展此前仅项目内清理的范围。不得据此删除其他用户文件或活动daemon/editor的socket、锁和临时依赖。旧独有实验材料先归拢到新临时目录的私有归档，记录位置；不把临时目录当唯一研究证据存储，也不批量改动旧实验源码中的历史路径。
