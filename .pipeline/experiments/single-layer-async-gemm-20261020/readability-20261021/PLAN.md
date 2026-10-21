# M39 当前入口精简（沿用原任务）

- [x] 核对真实调用链与无引用函数。
- [x] 直接绑定参数；移除两个无引用设备函数；普通forward不构造完整debug字典和资源统计。
- [ ] 构建及9真实签名，普通/debug接口、原Torch输出/有效KV/输入/RNG核验。
- [ ] 两独立进程、三KV旧新完整层Graph计时；无稳定退化后安装。
- [ ] 更新带读说明、HW与GitHub。

设备算术、tile、stage、同步不改。调用失败的CUDA错误传播保留；baseline/投机/原tile/保护文件不动。
预算和终态见JOB.json/STATUS.json；GPU最多900秒，累计原2483.115753/36000秒，CPU构建≤1800秒，新增≤1GiB；退出仅清理owned PID及出生身份。
