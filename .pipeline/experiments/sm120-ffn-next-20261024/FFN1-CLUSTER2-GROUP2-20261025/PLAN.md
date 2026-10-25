# FFN1双K64交接有限实验

- [x] Session聚焦WAM；读取实际S/旧A源码与等待证据。
- [x] 登记一个候选：MMA顺序、tile、1T4M3E及FFN2不变，两个K64的A连续16KiB共享；源/目的credit与ACK减半。
- [x] 编译独立normal/profile，原S/W/旧A/L二进制保留。
- [x] 27张量/七段BF16/输入KV/RNG/计数及两进程三KV完整层测量。
- [x] 输出18阶段表、采用决定、源/结果回传和owned清理。

预算60分钟/6500调用/2GiB/CPU4，见JOB。两槽仍同样大小；每组先等两个本地empty，rank1先arm16KiB事务再发credit；rank0等待两份A的TMA完成后发送，rank1完成DSM才ACK。rank0在ACK和本地empty均完成前不覆盖；rank1消费两槽后才发下一组credit。没有循环依赖。缺点：不再step+2提前回填，预先登记为可能负收益，不保证更快。整个FFN与完整block均比较，同轮原Torch无Graph。

终态：COMPLETE_NEGATIVE_RETAIN_S。数值通过、完整层三KV退化，停止候选。4842调用，无GPU失败/超时，IO恢复与owned退出核验完成。
