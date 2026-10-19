"""按物理GPU身份查询，避免把另一张卡的进程算作本作业争用。"""
import subprocess


def resource_for_gpu(uuid):
    def query(fields, apps=False):
        flag = '--query-compute-apps=' if apps else '--query-gpu='
        return subprocess.check_output(
            ['nvidia-smi', flag + fields, '--format=csv,noheader,nounits'],
            text=True, timeout=10).strip().splitlines()

    rows = [r for r in query('uuid,memory.used,memory.free,utilization.gpu')
            if r.split(',')[0].strip() == uuid]
    if len(rows) != 1:
        raise RuntimeError('registered GPU UUID missing or ambiguous')
    pids = [r.split(',')[1].strip() for r in query('gpu_uuid,pid', apps=True)
            if r.split(',')[0].strip() == uuid]
    return dict(gpu=rows[0], pids='\n'.join(pids))
