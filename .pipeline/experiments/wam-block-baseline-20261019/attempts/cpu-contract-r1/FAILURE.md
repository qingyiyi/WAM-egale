# 首轮CPU契约预检失败

14项中4项ERROR；同一原因：`AttributeError: WanAttention object has no attribute attn_mode`。发生于独立B0的前置检查，尚未进入模型数学差异判定。原reference未修改；修正仅把检查放回真实拥有字段的block。原工具session22123以exit1结束。首轮未采到PID出生身份，标NOT_CAPTURED，不补造。
