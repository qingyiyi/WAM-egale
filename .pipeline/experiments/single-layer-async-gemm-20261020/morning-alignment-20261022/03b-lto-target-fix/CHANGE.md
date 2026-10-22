# 第三候选构建修复

03在CPU阶段1.666秒退出；nvcc报-dlto与Torch显式-gencode冲突，0GPU。按同一官方文档，使用code=lto_80生成IR，dlink已有-dlto/sm80。保持同一第三候选、原数学与资源边界；不是第四种性能配方。旧FAILED保持。
