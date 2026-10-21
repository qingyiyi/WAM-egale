# 首次集成编译失败

context Storage增加stride模板后，gemm_pipeline.cuh旧generic dispatch仍按stride128调用。编译器拒绝参数不匹配，未启动GPU。修复该调用为136，其他FFN/action路径不变。CPU17.848642秒、GPU0秒计入本轮。旧STATUS/log/source manifest保留，r2沿同预算与绝对截止执行。
