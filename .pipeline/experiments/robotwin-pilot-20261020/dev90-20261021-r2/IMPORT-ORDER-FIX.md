# 启动导入顺序修复

原dev90入口把Runtime导入放在common.setup之前，server_math所需mesh_helper尚未绑定，加载模型前退出。原记录保留，0teacher/0完整episode，未消耗任何固定case。

新入口保持common.setup先绑定原始helper，再导入EvalRuntime。与已验证collection_server相同顺序。单独helper仅分离定义，不改TF/target数学、tau、三臂、初态和审计。新命名空间尚未启动；原协议不自动重试，待本次明确批准。
