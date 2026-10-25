# 首个模型准备失败与修复

server01/panel01在任何teacher forward和episode之前退出：inspect.getsource对EvalRuntime生成的AST函数使用旧co_firstlineno，报告lineno is out of bounds。此前M41已经修复，但迁移入口取用了缺少这段修复的版本。本次复用M41 raw/FINAL-SOURCE/server_main.py中完整annotate函数，保留OUT范围检查及唯一函数AST确认，不改模型/调度/阈值数学。

旧失败保留；panel02/server02是同预算内准备修复后的新namespace，不重跑已执行episode。原21:56:53 deadline不变。
