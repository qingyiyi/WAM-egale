真实36 arm-query行、216个测量replay、36个保存输出CPU逐字节/finite核验全部通过；410 intents含36 capture，设备等价374。原同shapeB2/B14各61结果缓存及原始oracle独立认证，全部数值/完整KV/RNG/缓存值/版本守卫和清理通过。

下表为两first/last的ΣCUDA中位数，均为同一进程固定输入；百分比为缓存或引擎延迟下降，不是机器人性能。

|KV档|单状态缓存下降|native B14缓存下降|guarded B14缓存下降|cached native→guarded下降|cached guarded/C1成本比|
|---|---:|---:|---:|---:|---:|
|short|20.554%|53.481%|54.390%|3.861%|1.569|
|middle|15.880%|50.217%|51.794%|6.452%|1.288|
|long|12.648%|46.722%|51.615%|18.402%|1.010|
|all_equal_fixture|15.759%|49.973%|52.566%|10.274%|1.238|

缓存payload共3,070,230,528B（2.859375GiB），另实际prompt clone 67,108,864B（64MiB）；peak/首填wall及CUDA见JSON。不是单纯payload增量与全进程peak的因果分解。

不包含生产输入更新、缓存失效/生命周期、Graph重建和首填摊销；没有cached G1单独控制。强单状态与B14共同优化后的局部成本比不能替代ΣT/Σg、闭环SR或完整chunk/e2e加速。
