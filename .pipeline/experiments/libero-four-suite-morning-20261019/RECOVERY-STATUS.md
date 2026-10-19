# Object最新接续证据

## 09:19–09:21接续实证

固定14局已全部补齐，CURRENT_DATA原30局并集终态完成且guard=CLEAN。PACK_CURRENT原准入通过：FIT6164行、CAL3080行，all_episodes_terminal/all_rows_reached=true，counterfactual_columns_used=false、extra_teacher_forwards=0。ON3000已按固定500updates训练成功返回，控制器已进入RUNNING_OR_QUEUED_ALL_REMOTE_OBJECT500；FINAL_200_274已出现真实完整评估终态，不只是创建目录。

09:21巡检Matched Target为18/200，FINAL_200_274为5/225局（三臂合计，尚非5个共同完成组）；当前每卡一个模型worker，其他分片仍在队列。GPU工具报告PID与容器READY/OWNED的PID不相等，额外槽的observed.issubset(allowed)未通过；尚无可靠映射，不把未知PID标成自有、不放松原准入。PID-namespace只是待核机制假设，不能据此宣称发现他人进程。现有完整episode继续，不重启/截断。

按已有两套质量运行单局约60–73秒、当前两worker及阶段依赖，Object剩余墙钟仅能粗估半天至一天；尾部失败/更多并发会改变估算。原10/21绝对截止不变，到期不足则如实PARTIAL。所有并行计时仅探索性，四项false未改变。最新源指针：RECOVERY-PACK-TRAIN-SUMMARY.json、REMOTE-LATEST-AT-REPORT.json、REMOTE-WATCH-AT-REPORT.json、REMOTE-SLOT-ADMISSION-CHECK.json、PID-NAMESPACE-SOURCE-CHECK.txt。
