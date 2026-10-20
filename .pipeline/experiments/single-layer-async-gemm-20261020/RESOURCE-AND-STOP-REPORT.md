# M39资源与终态

## OBSERVED

T0北京时间2026-10-20 10:31:32.306794；登记停新11:26:32、清理11:31:32。实际10:35:57完成，生命周期264.838秒，低于3600秒上限。包含CPU构建168.288秒；四GPU worker进程生命周期合计96.029秒。后者仍含初始化/审计，不等于CUDA active时间；CUDA active累计UNKNOWN。

| 子进程 | 生命周期秒 | exit | timeout | PID | birth ticks |
|---|---:|---:|---|---:|---:|
| build-1 | 168.288 | 0 | False | 639723 | 706195094 |
| gate-1 | 13.835 | 0 | False | 640043 | 706211934 |
| timing-1 | 32.426 | 0 | False | 640118 | 706213323 |
| timing-2 | 32.926 | 0 | False | 640600 | 706216575 |
| stages-1 | 16.842 | 0 | False | 641130 | 706219879 |

675次完整block：gate99、timing各261、stages54；432计时样本、18Chrome traces、0机器人/训练/30层。远端新增文件11200579字节（约10.68MiB，STATUS时点），低于512MiB；CPU归档后少量JSON另存。父进程639721/birth706195083及五子进程均按身份复查已退出，GPU上下文空，两卡当时0MiB/0%，未向他人进程发信号。[清理](remote-evidence/POST-CLEANUP.json)、[原始终态](remote-evidence/STATUS.json)。本机保护清单差异为空，远端全部已列保护项不变。

## FAILED / UNKNOWN：预算字段不一致

归档发现本轮PLAN在执行前登记900次完整block，runner也硬编码900；但原始REGISTRATION.json误沿用M38字段full_block_cap=240及primitive_cap=600。实际675超过机器登记240达435次。因此资源审计记`FAILED_REGISTRATION_CAP_MISMATCH`，**不能报告所有登记预算均通过**。保留原registration、源hash和COMPLETE计算终态，另存[审计](RESOURCE-AUDIT.json)，不改成900、不回填旧记录。这是执行登记实现错误，当前已停止GPU工作，没有用剩余书面额度继续实验。

GPU子进程没有timeout/非零退出。后续可选CPU反汇编在归档前因cuobjdump不存在失败，未执行SASS核查，未新增GPU调用；[原失败](remote-evidence/POST-ARCHIVE-ATTEMPT1.json)、[UNKNOWN结论](remote-evidence/INSTRUCTION-AUDIT.json)。随后仅CPU归档成功；不是重跑GPU实验。NCU计数、SASS指令数量、实际CUDA active总和为UNKNOWN。

## STOP / RECOMMENDATION

本轮停止于有限性能改善，原Torch逐位9/9失败，禁止生产集成/正式质量性能放行。四科学权限false，旧失败不修改。GPU生命周期剩余上限约3335.2秒只是算术余额，不表示已排队或可重启本作业。所有GPU程序已清理。

用户最新要求后续直接改optimized、不另建登记；用已有记录保留最小必要成本/失败证据，代码与记录中的预算必须同源，避免再次复制旧常量。baseline/投机/原tile保持。更大tile尚未执行，不能称已获收益。
