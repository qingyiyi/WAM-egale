只更改B14 FFN1为M128×N256×K64、三级、8个MMA warp的原CUTLASS核心，共112tile；不改K归约、BF16/bias及其他阶段。代价：整层dynamic shared增大并可能降低L1容量，必须以三KV整层净收益选择。九签名逐位门不通过立即停；不因为stage变快掩盖全层变慢。
