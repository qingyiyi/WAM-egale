# pair406/Target 单局补齐提案（未授权、未启动）

原登记 EVAL100-PLAN.md 第16行规定：资源/基础设施失败记PARTIAL，不自动重试或删掉。pair406 的 C/P 已完整自然失败，Target 因VAE CUDA OOM中断；只恢复了未开始队列，没有自动重跑该T。

如用户明确同意，仅在原GPU1（GPU-4acf84ff-31ce-dead-c2a4-2a122b123b8c）新独立namespace补齐 `press_stapler / valid067 / T` 一条完整episode。使用原固定INPUT-MANIFEST中的同一case，冻结目标权重、Target路径、reset bank、种子、物理/图像/命令处理与官方horizon400；不重跑C/P，不当新独立样本。

上限：完整局1800秒、teacher尝试8000次、准备与清理合计900秒，新增持久化≤256MiB，全部计原160GPU小时及更早10/29 guard截止，不增加采购或总额。开始前核显存余量并预留整局/清理资源；不满足则不启动。一次尝试，无自动重试。CPU沿原server/client各2线程。

保留原OOM/已执行64命令/PARTIAL与C/P两条自然失败。新Target的完整终态、初始物理/reset/双相机/RNG与同组旧原件核验、原C/P首两S通过证据形成独立恢复收据；只有所有身份与审计通过，才能把三臂并为该完整组。不得改旧FAILED或原统计身份。

在未获同意前，该提案不进入自动watch的重跑队列。其余固定队列和已授权pair281/C补齐继续。
