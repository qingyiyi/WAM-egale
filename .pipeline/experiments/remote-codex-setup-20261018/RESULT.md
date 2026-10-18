# 新机资源比较与 Codex 安装结果

**后续用户决定：不用迁移，由用户自行结束遗留本机模拟器。以下资源比较与安装为已完成事实，原迁移建议不再执行；不停止既有远端评测。**

结论：建议把开发、编译、CPU 分析及后续工作主入口放在 BAAI 新机。当前 CPU 余量明显较大；尚未测量迁移后的工作负载加速倍数。GPU 单卡型号相同，整机资源与软件环境不同。

## 已观察事实

硬件/负载快照为北京时间 2026-10-18 22:29；完整输入见 LOCAL-INVENTORY.json、REMOTE-INVENTORY.json、COMPARISON.json。

| 项目 | 原服务器 | BAAI 新机 |
|---|---|---|
| CPU | Xeon Gold 6133，可用 16 个逻辑 CPU | EPYC 7742；可见 256 线程，但 cgroup 总配额仅 45 核等效量 |
| 2 秒 CPU 使用采样 | 约 15.96 核等效量，16 线程接近满载 | 约 4.74 核等效量，占 45 核配额约 10.54% |
| CPU 等待压力 PSI some avg10 | 约 94.9% | cgroup 约 1.2% |
| 内存 | 系统可见 62.76 GiB、MemAvailable 31.10 GiB | cgroup 上限 180 GiB、当前用量约 87.36 GiB（含缓存） |
| GPU | 1×A100-SXM4-40GB，400 W | 2×A100-SXM4-40GB，各 400 W |
| 驱动 | 595.71.05 | 535.161.08 |
| GPU 现场 | 约 26.1 GiB 已用、75% 利用率 | 两卡各约 28 GiB 已用、100% 利用率，既定评测继续 |

PSI 是存在 CPU 等待的时间比例，不是 CPU 利用率。45 核是聚合配额，不代表独占 45 个物理核。远端 /proc/meminfo 显示的约 1 TiB 是宿主信息，不能当作当前任务可用内存。共享盘空余约 12.64 TiB 也不等于个人额度或持久化保证。

## 安装交付

新机 PATH 及检查的项目安装目录原先没有 Codex/Node/npm；未读取任何凭据。复用原服务器已安装的完整原生发行目录，安装了 **codex-cli 0.162.1**，保留原生依赖、资源和布局，无需 Node.js。官方安装脚本下载返回 HTTP403，未执行该脚本；实际安装来源与方法见 INSTALL-SOURCE.json。

- 启动入口：`~/cxz/bin/codex`。
- 原生发行目录：`~/cxz/tools/codex/releases/0.162.1-x86_64-unknown-linux-musl`。
- `CODEX_HOME`：`~/cxz/.codex`，认证文件使用该目录的 file store。
- 缓存、状态、配置及临时目录均通过入口绑定到 `~/cxz`；该约束写入新的 Codex AGENTS.md。
- 没有修改系统安装目录、`~/.bashrc`、系统 PATH 或外部 `~/.codex`；没有迁移本机登录凭据、SSH 私钥或会话。
- 版本、主帮助、登录帮助均正常退出；`login status` 返回 `Not logged in`，这是尚未认证，不是安装失败。

新机终端执行：

```sh
cd ~/cxz/WAM-egale
~/cxz/bin/codex login --device-auth
~/cxz/bin/codex
```

设备登录需用户在浏览器完成。启动入口有范围检查，但 AGENTS.md 不是操作系统级隔离。新机使用说明另存 `~/cxz/tools/codex/README.zh-CN.md`。

安装包 452,470,413 字节（约 431.51 MiB）；SSH 传输/安装生命周期约 123 秒，无 GPU 分配、模型调用、模拟器、研究 smoke 或独立 SHA。完整输出见 REMOTE-INSTALL-RECEIPT.json；清理见 REMOTE-FINAL-AUDIT.json、LOCAL-CLEANUP.json。安装进程退出，原评测与 CPU 健康巡检未停止。

## 建议与未知

1. 优先把编辑、编译、统计与 CPU 密集工作移到新机，降低本机 CPU 排队对开发和 GPU 提交的影响。当前资源快照支持该建议，未证明特定工作提速多少；两款 CPU 的单线程速度未测。
2. GPU 的型号、显存和功耗上限一致，但数量、驱动、任务占用不同，不能称整机性能完全一致。正式性能仍需同一机器/环境、无争用窗口；不要混算两机时间或直接将旧本机数据移作新机结果。
3. 新机两卡正在跑 Goal/Spatial，Object 仍按既定队列等槽；不为安装或迁移强停它们。当前 GPU 剩余资源不是额外正式计时窗口。
4. 新机已有 WAM 评测工作树、模型与 HW；核查时尚无刚完成的 `operator_torch_alignment_v2/integration.py`。最新算子与最新 HW 尚未完整迁入，后续应使用独立目录，不覆盖运行中入口。此次完成的是 CLI 安装，不冒充全部开发工作或会话已迁移。

官方依据：[Codex CLI 安装](https://learn.chatgpt.com/docs/codex/cli)、[认证与 headless 登录](https://learn.chatgpt.com/docs/auth)。项目科学默认、旧失败及四项质量权限均不变。
