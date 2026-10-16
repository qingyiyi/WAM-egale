# TF64 LIBERO 试验适配器

这里仅封装训练好的双候选 Transformer draft 的加载和 `predict`，不修改现用 A_off、server、verifier、阈值或 scheduler。权重是旧 DEV 六门未通过的诊断模型；本目录不能作为部署依据。模型只接收当前紧凑动作状态、上一方向、`sigma`、`next_sigma`、`timestep`，没有图像、语言 prompt 或 cross-attention。若要研究 prompt cross-attention，需要另定义输入接口、训练数据和权重。

对外接口：

```python
import sys
sys.path.insert(0, "/rjs/cxz/论文探索/投机采样/experiments")
from tf_libero_pilot_v1 import load_for_server

draft = load_for_server(checkpoint_path, expected_sha256, device, expected_channels)
directions = draft.predict(current_state, previous_direction, sigma, next_sigma, timestep)
# directions: [2, *current_state.shape]
```

`expected_sha256` 必须来自试验计划中预先固定的外部值；加载时重新哈希文件，并核验训练计划 SHA、架构、seed、1000 次更新、17 个 FP32 模型张量、四组归一化和有限值。`expected_channels` 可传 `7` 或七个紧凑动作通道 ID。`device` 支持 CPU/CUDA；模型始终 FP32 计算，输出回到输入 dtype。测试者应把它作为试验性 draft 注入独立的 LIBERO harness，并继续使用目标模型对全部候选状态验证；单独运行本目录不会启动 LIBERO。

两个已封存诊断 checkpoint：

| seed | 文件 SHA-256 |
|---|---|
| 20260922 | `e171542917063057ceed93abd90194577bd248ade3f3655f82918c316a343704` |
| 20260923 | `238d4348de6322f6a30d40aa1693ecbe92ccd62b6f1de6218dd7071bbc347110` |

CPU fixture 用旧 task0/1 init8 DEV 两侧共 12 行，在 FP32 和 BF16 下逐元素比较原 `Predictor.load().predict`，并对照当时封存的批量预测；另验证错误 SHA、元数据、几何、非有限值和张量形状均被拒绝。运行时不初始化 GPU：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /rjs/cxz/miniconda3/envs/va/bin/python -B \
  投机采样/experiments/tf_libero_pilot_v1/fixture.py
```

CUDA 路径已按同一 FP32 张量运算实现，但本目录的 CPU fixture 不证明 GPU 数值一致、速度收益、LIBERO 闭环成功率或轨迹长度；这些必须由独立试验测量。
