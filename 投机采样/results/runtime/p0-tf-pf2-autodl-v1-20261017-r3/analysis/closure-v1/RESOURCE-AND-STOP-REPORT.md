# 资源、终态与止损

OBSERVED：r3终态COMPLETE_RECORDED_SIX_PAIR_PANEL，十二完整机器人任务，runner生命周期1253.530秒（约20.89分钟）；从原授权11:11:41到r3终态累计7063.751秒（约1.962小时），含准备/等待/失败，未重置起点。并非从首次GPU active才计费。

三次已登记runner生命周期分别24.193/37.641/1253.530秒。r1 WAM配置模块导入冲突，在模型构造前失败；r2两client在环境初始化/transport前因LIBERO包路径失败，两个零环境步基础设施attempt，不能算算法失败或偷偷删除。修复source namespace后r3运行原固定面板，不按机器人结果挑重跑。旧r1/r2原FAILED/STOP/源码快照/报告均保留。

六个已申请server进程的登记到reap累计0.728764 GPU-worker生命周期小时（含两卡、初始化和失败准备；不是CUDA active时间）；实际全模型GPU active时间UNKNOWN。各CPU server2/client2，实际绑定0–7，分析单核12；没有draft DDP。

已计数完整episode target forward共16054，最大单局2676<4000。r2缺失COUNTER按原规则额外保守扣8000，未覆盖非episode调用另留1000；保守总账25054/49000、余额23946，不把UNKNOWN当0。原12局已完成，不因有余额扩任务。详见RESOURCE-DETAILS.json。

原23:11:41清理硬截止和22:56:41停新工作未变；实际13:09:25已清理作业。原始证据1267文件/55035793字节（约52.5MiB），远小于3GiB；迁移target权重24,359,646,412字节属于已批准准备资产，与新增研究证据分列。临时249MiB传输包已双端清理；本轮20.1MB证据传输包在完整回传/解包后也已双端删除，清理收据另留，原件保留。

两卡全部own进程组EMPTY；guard最终CLEAN，无遗留pins；后续真实资源观察85个登记身份无存活，nvidia-smi计算进程空、每卡1MiB/0%利用率。不是通过杀其他进程获得。实例关机NOT_REQUESTED_NOT_EXECUTED，租卡可能继续计费。

默认reset全臂使用同512-token BF16 bank，十二次耗时0.014952–0.107278秒；加载模型/首次bank构建单列。原阈值、旧FAILED、tile及491保护metadata不改。远端未运行smoke、独立测试或项目SHA；来源/大小/raw字节证明不冒充新的密码学复验。

止损：本轮投入NO_GO；不自动扩规模、补旧19/24项、训练、调tau、PF3/4、TEST或正式性能。时间余额不是追加实验授权。RECOMMENDATION：将下一个有界教师路径诊断交由新登记决定；10/22 go/no-go、10/23接口与claim冻结、10/28主表、11/01 18:00中文证据包的内部节点保留，证据不足缩主张。
