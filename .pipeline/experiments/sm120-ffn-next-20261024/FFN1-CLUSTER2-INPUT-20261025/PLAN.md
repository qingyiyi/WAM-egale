# FFN1双CTA共享A：接续用户新授权

当前保留S。共享B已完成负结果，保留原窗口/全部产物；本轮不重置它。

- [x] 已授权、5090空闲、独立S/W/B二进制核对。
- [x] 实现共享A与同映射独立加载对照L；源码/所有448tile与数值边界审查。
- [x] 正常/对照/观察器独立编译；三真实fixture逐位/七段/状态/实际计数门。
- [x] 两独立进程、原Torch/S/W/B/L/共享A，同场eager/noGraph完整层；阶段表。
- [x] 必要时等待位置观察器；数值通过后才用于诊断，扰动分列。
- [x] 回传/owned清理、HW和GitHub main、最终完整阶段表/采用决定。

100分钟含准备/等待/失败/清理，9000完整block调用，2GiB，4CPU，绝对截止见JOB。
224个cluster任务、每对相同M64/不同N128。rank0 TMA A8KiB+B16KiB，rank1仅B16KiB；8KiB DSM把A复制进对端既有local槽。输入双缓冲及ready/empty/credit/ACK协议沿用共享B，保留1T4M3E、完整K3072与FFN2七段递推。
软件tile请求global输入504→420MiB，另加84MiB DSM；不是HBM实测。
本轮结果与共享B同场比较，L区分映射和共享。不得预设共享A会快，不自动扩成多tile累计或完整DAG。

终态：共享A未采用，保留S；等待诊断、公开入口、回传/清理已闭合。GitHub发布以PUBLICATION.json为准。
