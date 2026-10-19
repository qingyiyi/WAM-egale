# M31执行完成：范围与剩余项

- [x] 登记、冻结baseline及依赖，真实输入引用。
- [x] optimized独立参数、native、C++与构建产物；默认sequential等价路径。
- [x] self/cross attention显式输出tile、原split/combine；实际三段Transformer子图。
- [x] FFN1显式tile可选；FFN2原cuBLASLt数值边界保留。
- [x] 编译、ELF/独立进程隔离、9组block/72状态/输入RNG/API门及真实dispatch。
- [ ] 新30层正确性：BLOCKED，当前可用4.9GiB不足约28GiB；下一阶段补齐。
- [x] 有限结果/错误/资源终态归档；旧结果不改。

本轮静态子图实现交付完成，剩余整模型门明确未过。没有persistent调度或新性能结论，不自动启动后继。结果与账本见RESULT.md、RESOURCE-AND-STOP.json。
