当前状态：CPU_PREPARATION_WAIT_REAL_S04_V2_COMPLETE。未创建正式运行PLAN，未运行GPU。

最初只有CPU contract、因S04 r1失败而停止的事实保留在
results/runtime/p0-wam-static-prompt-baseline-preparation-v1-20261016-r1/
PREPARATION-STOPPED.json。root随后授权在独立S04 v2观察纠错期间恢复本目录CPU准备；
该授权不代表S04通过，也不替代真实GPU门。S04 v2若失败，不执行本实验。

这是普通强基线核查，非新MegaTransformer贡献。固定三档KV各first/last真实query，
六臂：n1 dense cache OFF/ON，n7 native shared cache OFF/ON，n7 guarded cache OFF/ON。
n1与n7分别实际调用原B2、B14形状Torch投影首填，各保存完整text_embedder和30层cross
K/V共61个BF16结果；不折叠CFG、不用B14切片充当B2投影、Q/normK/attention/FFN等原样。
原prompt bytes必须在全部六query一致，模型/缓存/输入版本不变。首填时间和payload另列。

两次真实全forward首填 +12次原同shape oracle +36次全局eager准入
+72 warm +36 capture intents +36首replay +216六轮交错计时
=410 intents；capture不执行，设备等价374，硬帽500。
全部36次eager逐位/有限通过后才允许任何Graph。六轮使用固定平衡顺序，不挑最好轮。
每query每arm独立Graph/pool；先实际replay再验数值。RoPE GPU常量一直保留到所有Graph销毁。
每轮输出对原同shape oracle逐位，完整有效KV、metadata、输入/slots/cache值/版本/RNG不变。

缓存只复用原来已计算的同shape BF16结果。静态准入和Python守卫不在Graph replay执行，
不能宣称为生产缓存失效协议。n1采用既有graph_matrix相同数学的固定slots dense路径，
其动态nonzero和host元数据准入与n7一起移到静态准入；无减精度、无新kernel。
冷首填wall/CUDA时间不混入hot Graph时间；新输入、Graph构建、真实cache生命周期均未测。
不能用cached B14对uncached n1来声称公平投机收益，也不能外推真实chunk/SR/e2e。

先CPU审查，再由root在S04 v2完整六policy/cache、全部实际事件/50提交、逐位及自有清理
真实通过后显式prepare/inspect/execute-reviewed。资源单次600秒（540活动+60清理），
CPU7，独占双锁，22GiB初始/0.5GiB运行显存余量、128MiB输出、5GiB磁盘余量，无重试。
prepare在真实门之前不创建目录，不造S04证书，不跳过原v1 FAILED保留/累计资源检查。
