---
kind: decision
name: 允许补跑RoboTwin首局视频入口失败
level: constraint
scope: cycle:wam-research-20261016;robotwin_pilot;place_object_basket-valid00
status: active
sources:
  - 用户2026-10-20明确回答允许补跑，保留原PARTIAL
---
# 仅补跑首个视频入口失败case

用户允许在原预算和截止内补跑`place_object_basket/valid00`。首次attempt在动作前因可选视频导出入口未彻底关闭、找不到ffmpeg而退出，0teacher/0动作，原PARTIAL与耗时保留。

修正入口仅移除可选视频输出目录，不改相机观察、渲染、物理、动作或成功判据。与其余五条尚未尝试任务接续，沿原六任务清单及绝对deadline；不扩为普遍自动重试许可，不重置24h或首次六局预算。
