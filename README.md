# WAM-egale：FastWAM投机推理当前证据

2026-10-27公开快照，供网页版讨论。请固定commit；本地HW与原始实验目录仍是执行依据。

- [昨晚完整简报](review/fastwam-20261027/BRIEF.md)
- [复制给网页版的讨论Prompt](review/fastwam-20261027/WEB-DISCUSSION-PROMPT.md)
- [6000局与原始速度数据说明](review/fastwam-20261027/DATA-README.md)
- [当前代码导航](review/fastwam-20261027/CODE-MAP.md)、[版本和来源SHA](review/fastwam-20261027/MANIFEST.json)
- [HW当前快照](.pipeline/INDEX.md)、[历史入口](review/history-before-fastwam-20261027/README.md)

LIBERO四套三臂6000局完成：target97.50%、TF97.25%、TF+PF97.20%。共同prompt缓存5090动作1.150–1.242×、完整chunk1.057–1.080×，速度只统计原DEV三臂共同成功来源。Robo仅4局canonical工程检查通过，尚无投机全量。新增Cross-KV四观测对照提示更强共同基线会缩小投机收益；未合入主版本。

全量含FIT/CAL/DEV和历史暴露，不是独立TEST；原版闭环质量与新缓存速度分版本，重复运行路径差异仍未定因，四项科学权限均false。失败、长尾和负结果保留。不上传权重、teacher/相机/KV原tensor、DSO、凭据；公开源码不等于完整部署环境。仅main作为公开进度镜像，旧结果由历史commit追溯；不生成论文正文。来源和许可见既有NOTICE/LICENSE。
