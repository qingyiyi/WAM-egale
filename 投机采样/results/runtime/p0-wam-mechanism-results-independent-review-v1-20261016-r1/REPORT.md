实际18行eager与24行静态full30 Graph已核，保存的180个wall/CUDA中位数从各6样本重新计算一致。24行Graph均记录与原同shape输出逐位相同；来源221/222文件无漂移、9项大权重仅stat，两个作业GPU采样未见外来PID、自有进程全退出。

|KV|位置|7顺序B2 eager ms|B14 eager ms|原shared B14 ms|private shared eager ms|shared Graph ms|同shape Graph收益|
|---|---|---:|---:|---:|---:|---:|---:|
|short|first|495.61|91.35|74.63|71.22|68.17|1.045×|
|short|last|590.99|94.43|86.27|73.54|68.09|1.080×|
|middle|first|613.77|111.20|86.74|89.76|73.63|1.219×|
|middle|last|595.63|110.25|83.55|74.83|73.49|1.018×|
|long|first|605.81|147.09|87.10|80.69|79.01|1.021×|
|long|last|608.14|147.88|88.17|80.72|79.11|1.020×|

同shape dense n7 private eager→Graph仅1.014–1.069×，shared为1.018–1.219×（中KV首query离散较大）。7顺序B2 eager→B14 sharedGraph的7.27–8.68×混合batch/共享/hostguard移出/Graph，不能写成Graph单独收益。尚无实际相同7状态的整体顺序B2 Graph；7×首根单状态Graph不是实测。

三个完整chunk各50次scheduler更新，树数19/21/21，Σg48/49/48，平均2.526/2.333/2.286；加起始target更新2/1/2才各50。n7是候选树宽，绝非g。

真实ledger每query307个Linear（30block各10个+7个embedding）；305个输入BF16、2个time embed输入FP32。动作M224有245调用，文本M7168有62调用；2MNK估算文本占线性运算量81.0%，不是81%耗时。单attn1 out投影的NCU只能证明该M32/96/224算子机制，不能代表全部307调用。

eager作业159.90 GPU秒、输出2.636GiB；Graph93.78秒、输出约0.207MiB。静态Graph不计输入更新/重建/打包/生产cache生命周期。不同M或顺序/批量的BF16结果可能不同，不能推导接受路径或SR保证。

现有结果足够选择受限路线B kernel小样机：必须与同shape、同Graph、相同真实输入和原权重基线比较，并守住数值门；不足以启动大融合或声明全chunk/e2e收益。下一步先补真实7顺序B2整体Graph强baseline与NCU实际counter，再按算子时间份额决定扩展。

NCU r1包装入口pin失败，无任何真实counter；r2待终态增补。独审仅CPU2、0GPU/NVML/SSH、无HW或原FAILED改写。原REVIEW-ATTEMPT1保留：审计曾误假设307输入全BF16，真实为305+2，已按真实ledger纠正。
