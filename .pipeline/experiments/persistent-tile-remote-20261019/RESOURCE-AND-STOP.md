# M32资源与停止报告

本轮19:52:29北京时间开始；原工作停止22:47:29、清理22:52:29不变。用户20:32明确仅一层完整block，取消full30迁移/验证，不改写M31显存不足历史。

| 作业 | 生命周期秒 | 终态 | 完整block |
|---|---:|---|---:|
| block-state-1 | 16.837 | 自然exit0 | 207 |
| tile-coverage-1 | 14.885 | 自然exit0 | 20 |
| block-timing-1 | 12.781 | 自然exit1，PID命名空间误判 | 0 |
| block-timing-2 | 16.787 | 自然exit0 | 252 |
| block-timing-3 | 16.637 | 自然exit0 | 252 |

累计77.927/1800秒，731/900次完整block，另48次真实原语。预算余额1722.073秒与169次block仅是原预算余额，本轮收口后不自动转为后续实验授权。CUDA active时间UNKNOWN，不能用生命周期冒充。共享host/另GPU Object保留，正式E2E没有运行。

构建为CPU114.538秒；远端工具链独立CUDA12.4、Torch2.9.0+cu126。20:59现场核对所有记录owned的PID/出生时间均不再运行，GPU0显存0MiB；GPU1 Object仍在原任务。记录见FINAL-RESOURCE.json。没有杀他人任务、改驱动、服务或时钟；脚本退出不声称服务器/租卡实例关机。

准备失败和修正：

- stage/rsync因远端无rsync退出；未安装系统依赖。
- 首tar流太慢，中止自有连接，发送方SIGPIPE/接收255保留；后验出生身份标注，不伪造启动收据。
- SFTP复制工具链超时；后续完整工具链覆盖了未完成传输。
- 反向隧道工具链完成，工作区达到900秒传输上限而中止；随后用户收缩30层范围，原超时不是伪装成用户主动停止。隧道/HTTP均退出。
- 四并行压缩单层输入流全部完成；原5文件各自receipt保留。曾有本地Python语法拼接错误，在启动流前修正，无GPU工作。
- 计时guard误以宿主NVML PID应等于容器PID，0 block时失败。修正只作用于资源审计，原数学/精度门不变；新独立attempt2/3完成。原失败不覆盖。

停止方向：本轮取消完整30层；停止把未产生收益的stage-local persistent当默认加速。没有开展大规模调参、跨阶段融合、正式E2E或新机器人评估。约667,442,846逻辑字节远端隔离材料保留供单层研究；本机报告/原日志体积另列，不删除不可重建证据。

CUDA13问题：本机driver595.71.05，远端535.161.08；NVIDIA常规minor-compatibility表CUDA13最低580，故远端不能直接按普通CUDA13环境切换。本轮没有改环境到CUDA13；若日后考虑forward-compat包应另核验，不据此承诺可用。[官方兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。
