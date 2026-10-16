本轮完整50原始独立复核通过，原执行COMPLETE不等于候选质量通过。

TF按原预登记保留：False。未满足：['risk_no_new_TF_failure']。

| 分层 | arm | 成功/条数 | 全步数P50/P90 | 失败800均值 |
|---|---|---:|---:|---:|
|fresh_DEV|MLP_CONT|10/10|259.5/318.7|267.700|
|fresh_DEV|T|10/10|251.5/297.2|257.500|
|fresh_DEV|TF_CONT|10/10|257.0/316.4|265.000|
|risk_DEV|T|5/8|256.5/796.0|452.750|
|risk_DEV|TF_CONT|5/8|315.5/796.0|472.000|
|repeatability|T|4/4|322.5/489.8|359.750|

fresh TF/T共同成功长度比：{'n': 10, 'mean': 1.0285961011546938, 'p50': 1.0212123253829222, 'p90': 1.0748842592592593, 'min': 0.9826839826839827, 'max': 1.0925925925925926}。
风险四格：{'reference_0_candidate_0': 2, 'reference_0_candidate_1': 1, 'reference_1_candidate_0': 1, 'reference_1_candidate_1': 4}；T成功而TF失败：[{'task': 4, 'init': 26, 'reference_success': True, 'candidate_success': False, 'reference_steps': 249, 'candidate_steps': 796, 'both_success_steps_ratio': None}]。
28质量初始pair：物理/RNG 28/28；严格像素 25/28。2个target重复pair独立列示。
全部3184 PT/34776tensor与数组实际读取，finite与完整文件名/请求序列通过；61859target forwards。原RESULT/PAIRING逐字段重建一致；source 173项、两权重与原plan绑定通过。
自有清理/独立clock终态通过；GPU-job charged 8056.511s仅资源诊断，不是正式性能。

未混晨间80/历史50，未读取TEST。task7/init24未包含，不能称已回归。raw velocity未保存，未重算原score构造；仅一份每arm Torch初始RNG及每reset指纹，其他未保存内部状态UNKNOWN。相机差异不能推出失败根因；四权限保持false。
