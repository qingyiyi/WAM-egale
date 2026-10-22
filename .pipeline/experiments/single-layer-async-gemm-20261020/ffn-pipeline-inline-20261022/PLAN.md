# FFN外层函数内联单因素验证

- [x] 用户批准先移除外层流水函数保存恢复边界，再处理FFN1尾部。
- [x] 只更改action_ffn_pipeline内联属性；旧B作为内部回归参照，原Torch为外部baseline。
- [ ] CPU编译，检查寄存器/local/SASS。
- [ ] 9真实签名及36结构有限逐位/状态；失败不计时。
- [ ] 三KV两独立进程完整层，同等最终cross KV缓存。
- [ ] 独立FFN对照/资源与采用决定，清理并更新HW。

本轮上限60分钟，CPU2、远端GPU0，新存储768MiB；累计起点12603.609944/36000秒，不修改旧账本。四科学权限及production_admitted=false。GPU1 RoboTwin不占用。详见JOB.json。
