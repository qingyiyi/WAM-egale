"""One-shot single-worker controller, CPU gate first, detached-safe watchdog."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback

BASE = Path(__file__).resolve().parent.parent/'single_worker_chc_gpu_v4'
import sys
if str(BASE) not in sys.path:sys.path.append(str(BASE))

from task_contract import OWN, ROOT, REVISION, ENABLED, APPROVED, COUNTER_JOB, STAGES, Ledger, enabled_plan, counter_identity, freeze, stamp, write, digest
from resource_guard import NVML, Idle
from counter_recovery import recover_owned, budget_from_recovery
import ownership


class Supervisor:
    def __init__(self,job):
        self.job=Path(job).resolve();self.p=enabled_plan(job);self.children=[];self.pins={}
        self.lock=threading.RLock();self.stop=threading.Event();self.done=threading.Event()
        self.worker=None;self.nvml=None;self.gpu_start=None;self.gpu_end=None
        self.stage='CPU';self.passed=[];self.ledger=Ledger();self.index=0;self.timed_out=False
        self.failure_reason=None;self.counter_recovery=None

    def read_counter(self):
        parent=self.ledger.summary()
        if self.gpu_start is None:return dict(status='GPU_NOT_STARTED',ledger=None)
        owner=None
        if self.worker is not None:
            pin=self.pins.get(self.worker.pid)
            if pin is not None:owner=dict(pid=pin.pid,uid=pin.uid,start_ticks=pin.start_ticks)
        try:
            row=recover_owned(self.job/'worker/counter.jsonl',counter_identity(self.job),
                              expected_owner=owner,parent_last_ack=parent)
        except Exception as exc:
            row=dict(status='REJECTED_COUNTER',ledger=None,error=repr(exc),
                     parent_last_ack_budget=parent,parent_ack_is_not_final_counter=True,
                     whole_stage_success_inferred=False,raw_journal_modified=False)
        self.counter_recovery=row;return row

    def budget_summary(self):
        if self.gpu_start is None:
            return {**self.ledger.summary(),'counter_source':'no_GPU_attempt'}
        return budget_from_recovery(self.ledger.summary(),self.read_counter())

    def status(self,**kw):
        write(self.job/'status.json',dict(state=self.stage,updated_at=stamp(),supervisor_pid=os.getpid(),
            GPU_started=self.gpu_start is not None,GPU_elapsed_sec=None if self.gpu_start is None else (self.gpu_end or time.monotonic())-self.gpu_start,
            budget=self.budget_summary(),passed_gates=self.passed,training=False,not_tile=True,**kw))

    def guard(self):
        if self.stop.is_set() or (self.job/'STOP_REQUESTED').exists():raise RuntimeError(self.failure_reason or 'owned stop requested')
        if self.gpu_start is not None and self.gpu_end is None and time.monotonic()>=self.gpu_start+2160-15:
            self.expire('36min deadline with15s cleanup reserve');raise TimeoutError('GPU safety deadline')
        if shutil.disk_usage(self.job).free<20*2**30:raise RuntimeError('disk below20GiB')

    def environment(self,*,cpu=False):
        e={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_','NSYS_'))}
        e.pop('CUDA_LAUNCH_BLOCKING',None);ref=self.p['frozen_reference']
        e.update(dict(CUDA_VISIBLE_DEVICES='' if cpu else '0',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',
            OMP_NUM_THREADS='7',MKL_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',NUMEXPR_NUM_THREADS='7',
            P0_SHARED_PREFIX_V3_ENABLED='1',P0_SHARED_PREFIX_V3_GPU_APPROVED='1',
            LINGBOT_SEED='20260922',LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',
            LINGBOT_VA_LIBERO_SUITE='libero_10',LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'],
            WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',WAM_P0_ACTION_DRAFT_MODE='off',
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=self.p['frozen_A_off'],WAM_P0_ACTION_DRAFT_SELECTOR='global',
            WAM_P0_ACTION_DRAFT_THRESHOLD=str(self.p['threshold']),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(self.p['ports'][0]),
            RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PYTHONPATH=str(OWN)+':'+self.p['worktree']+':'+str(ROOT/'投机采样/scripts')))
        e['PYTHONPATH']=str(OWN)+':'+str(BASE)+':'+e['PYTHONPATH']
        e['TRITON_CACHE_DIR']=str(self.job/'triton-cache')
        if not cpu:e[COUNTER_JOB]=str(self.job)
        return e

    def event(self,name,result):
        path=self.job/'events'/(name+'.json')
        write(path,dict(job=str(self.job),event=name,timestamp=stamp(),GPU_started=self.gpu_start is not None,
            budget=self.budget_summary(),passed_gates=self.passed,result=result),exclusive=True)
        with (self.job/'HW-events.log').open('a') as log:
            subprocess.run(['flock','-w','60',str(ROOT/'.pipeline/.kv-hw-append.lock'),'node',str(OWN/'hw_event.mjs'),str(path)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
        self.status(last_event=name,next_step=result.get('next_step'))

    def spawn(self,cmd,log,*,cpu=False,gpu=False):
        with self.lock:
            self.guard()
            if gpu:
                if self.worker is not None or self.gpu_start is not None:raise RuntimeError('one model worker/one attempt')
                self.gpu_start=time.monotonic();self.ledger.deadline=self.gpu_start+2160-15
                write(self.job/'GPU-ATTEMPT.json',dict(timestamp=stamp(),attempt=1,
                    job=str(self.job),execution_revision=REVISION,
                    monotonic_start=self.gpu_start,monotonic_deadline=self.gpu_start+2160,
                    timer_includes_loading_capture_tests_profile_and_cleanup=True),exclusive=True)
            Path(log).parent.mkdir(parents=True,exist_ok=True)
            with Path(log).open('ab') as out:
                child=subprocess.Popen(cmd,cwd=self.p['worktree'],env=self.environment(cpu=cpu),stdin=subprocess.DEVNULL,
                    stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            self.children.append(child)
            ownership.pin_roots([dict(pid=child.pid,process_group=child.pid,command=cmd)],self.pins)
            if gpu:self.worker=child
            write(self.job/'owned-processes.json',[dict(pid=x.pid,process_group=x.pid,command=x.args) for x in self.children])
            if gpu:self.start_watchdog()
            return child

    def cleanup(self):
        # Never signal a PID/group solely because a stale integer matches.
        with self.lock:
            for sig in [signal.SIGTERM,signal.SIGKILL]:
                for child in reversed(self.children):
                    try:
                        info=ownership.read_process(child.pid);root=self.pins.get(child.pid)
                        if root is not None and (info.start_ticks,info.uid,info.pgid,info.argv)==(root.start_ticks,root.uid,root.pgid,root.argv):os.killpg(child.pid,sig)
                    except (FileNotFoundError,ProcessLookupError):pass
                for child in reversed(self.children):
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:pass

    def expire(self,reason):
        self.failure_reason=reason;self.timed_out|='deadline' in reason;self.stop.set()
        (self.job/'STOP_REQUESTED').write_text(reason+'\n');self.cleanup()

    def start_watchdog(self):
        def watch():
            while not self.done.wait(2):
                try:
                    if self.gpu_end is not None:return
                    if time.monotonic()>=self.gpu_start+2160-15:self.expire('36min independent watchdog deadline');return
                    sample=self.nvml.sample()
                    apps='\n'.join(str(v['pid'])+',0' for v in sample['compute']+sample['graphics'])
                    with self.lock:foreign=ownership.foreign_pids(apps,self.pins)
                    with (self.job/'GPU-ownership-samples.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=stamp(),sample=sample,foreign=foreign))+'\n')
                    if foreign:self.expire('foreign GPU process detected: '+str(foreign));return
                    if sample['memory_free_bytes']<8*2**30:self.expire('GPU free memory below8GiB');return
                    if shutil.disk_usage(self.job).free<20*2**30:self.expire('disk below20GiB');return
                except Exception as exc:self.expire('watchdog inspection failed: '+repr(exc));return
        threading.Thread(target=watch,name='single-worker-safety-watchdog',daemon=True).start()

    def cleanup_audit(self):
        self.cleanup();remaining=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                info=ownership.read_process(int(proc.name))
                if ownership.is_owned(info,self.pins):remaining.append(info.pid)
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        occupied=[]
        for port in self.p['ports']:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try:s.bind(('127.0.0.1',port))
                except OSError:occupied.append(port)
        sample=self.nvml.sample() if self.nvml is not None else None
        if sample:
            remaining.extend(v['pid'] for v in sample['compute']+sample['graphics'] if v['pid'] in self.pins)
        out=dict(status='PASS' if not remaining and not occupied else 'FAIL',remaining_owned_pids=sorted(set(remaining)),
            occupied_owned_ports=occupied,other_processes_not_signalled=True,sample=sample)
        write(self.job/'OWNERSHIP-CLEANUP.json',out)
        if out['status']!='PASS':raise RuntimeError('owned cleanup incomplete')
        if self.gpu_start is not None:
            recovery=self.read_counter();out['counter_recovery']=recovery
            write(self.job/'RECOVERED-COUNTER.json',recovery)
        if self.gpu_start is not None:self.gpu_end=time.monotonic()
        self.done.set();return out

    def cpu(self):
        write(self.job/'CPU-ATTEMPT.json',dict(timestamp=stamp(),pid=os.getpid()),exclusive=True)
        self.stage='CPU';self.status();freeze(self.p)
        tasks=[('test_prefix_cpu.py',['--out',str(self.job/'preflight/prefix-tests.json')],'prefix-tests.log'),
               ('test_entry_cpu.py',['--job',str(self.job),'--out',str(self.job/'preflight/entry-tests.json')],'entry-tests.log'),
               ('worker.py',['--job',str(self.job),'--cpu-probe'],'worker-probe.log')]
        for script,args,log in tasks:
            child=self.spawn([self.p['model_python'],str(OWN/script),*args],self.job/'preflight'/log,cpu=True)
            if child.wait(timeout=900)!=0:raise RuntimeError('fresh shared-prefix CPU preflight failed: '+script)
        rows=[json.loads((self.job/'preflight'/name).read_text()) for name in ('prefix-tests.json','entry-tests.json','worker-CPU-probe.json')]
        if any(r['status']!='PASS' or r.get('CUDA_initialized') or r.get('CUDA_attempts') for r in rows):raise RuntimeError('fresh CPU result rejected')
        probe=rows[-1]
        if probe['production_model_module']!='modules.model':raise RuntimeError('real production loader probe missing')
        if probe['recovered_counter']['status']!='CLOSED' or probe['recovered_counter']['ledger']['target_attempted']!=58:raise RuntimeError('same worker persistent CPU counter missing')
        if (self.job/'worker/ready.json').exists() or (self.job/'GPU-ATTEMPT.json').exists():raise RuntimeError('CPU contaminated GPU')
        freeze(self.p)
        g=dict(status='PASS',plan_sha256=digest(self.job/'plan.json'),implementation_sha256=self.p['implementation_sha256'],
            CPU_tests=sum(r.get('tests_run',0) for r in rows),same_actual_worker_CPU_probe=True,
            production_loader_all30=True,CUDA_initialized=False,new_GPU_started=False,
            strict_bitwise_GPU_acceptance_not_relaxed=True,diagnostic_nonbitwise_approved=self.p['diagnostic_nonbitwise_approved'])
        write(self.job/'preflight/CPU-GATE.json',g,exclusive=True)
        self.passed.append('CPU');self.stage='CPU_READY'
        self.event('CPU',{**g,'next_step':'新job CPU关通过；按本轮授权检查独占空闲资源，之后只启动一次单worker GPU。'})

    def resource(self):
        self.stage='RESOURCE';self.status();self.nvml=NVML();idle=Idle();announced=False;idle_start=None
        # Explicit user authorization: read-only poll every180s until empty.
        # No model attempt/timer/VRAM claim exists during this resource queue.
        while True:
            self.guard();sample=self.nvml.sample()
            with (self.job/'resource-samples.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=stamp(),sample=sample))+'\n')
            if sample['compute'] or sample['graphics'] or sample['memory_used_bytes']:
                idle=Idle();idle_start=None;self.stage='WAITING_RESOURCE'
                next_step='按用户授权每3分钟只读检查，空闲后稳定独占确认再自动启动一次GPU；0本任务GPU，不信号外部PID。'
                if not announced:
                    self.event('WAIT',dict(status='WAITING_RESOURCE',sample=sample,poll_interval_sec=180,next_step=next_step));announced=True
                else:self.status(next_step=next_step)
                # Event.wait allows termination without uninterruptible sleep.
                self.stop.wait(self.p['resource_poll_interval_sec']);continue
            self.stage='RESOURCE';self.status()
            if idle_start is None:idle_start=time.monotonic()
            if idle.add(sample,time.monotonic()):
                if sample['memory_total_bytes']!=40*2**30:raise ValueError('expected40GiB A100')
                mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
                if mem['MemAvailable']<10*2**30:raise RuntimeError('CPU available memory below10GiB')
                for port in self.p['ports']:
                    with socket.socket() as s:s.bind(('127.0.0.1',port))
                freeze(self.p);write(self.job/'RESOURCE-GATE.json',dict(status='PASS',sample=sample,stable_samples=15,
                    CPU_available_bytes=mem['MemAvailable'],disk_free_bytes=shutil.disk_usage(self.job).free),exclusive=True);return
            if time.monotonic()-idle_start>=180:raise RuntimeError('empty GPU idle utilization not stable within180s;0 GPU attempt')
            self.stop.wait(1)

    def wait(self,path,timeout=600):
        deadline=time.monotonic()+timeout
        while not Path(path).exists():
            self.guard()
            if self.worker.poll() is not None:raise RuntimeError('worker exited before result')
            if time.monotonic()>=deadline:raise TimeoutError('bounded stage timeout')
            time.sleep(.05)

    def command(self,stage,op,**args):
        self.guard();self.ledger.begin(stage);self.stage=stage;self.status()
        write(self.job/'budget.json',self.ledger.summary())
        path=self.job/'worker/inbox'/f'{self.index:04d}.json';out=self.job/'worker/outbox'/path.name
        self.index+=1;write(path,dict(id=self.index-1,job=str(self.job),stage=stage,op=op,**args),exclusive=True)
        self.wait(out,715 if op in ('capture','diagnostic') else 300)
        row=json.loads(out.read_text());actual=row.get('ledger',{})
        recovery=self.read_counter()
        if recovery.get('ledger')!=actual:raise ValueError('worker ack differs from persisted owned counter')
        self.ledger.attempted=actual.get('target_attempted',self.ledger.attempted)
        self.ledger.completed=actual.get('target_completed',self.ledger.completed)
        self.ledger.stage_attempted=actual.get('stage_attempted',0)
        self.status()
        if row.get('status') not in ('PASS','PASS_KERNEL_DIAGNOSTIC'):raise RuntimeError('worker gate failed: '+row.get('error','unknown'))
        if actual['target_reserved']!=self.ledger.reserved or actual['target_attempted']>715:raise ValueError('worker/controller budget mismatch')
        write(self.job/'results'/(stage+'.json'),row,exclusive=True);return row

    def gpu(self):
        self.p=enabled_plan(self.job,cpu_gate=True);self.passed=['CPU'];self.resource()
        self.stage='GPU_LOADING'
        self.spawn([self.p['model_python'],str(OWN/'worker.py'),'--job',str(self.job)],self.job/'worker/model.log',gpu=True)
        self.event('START',dict(status='PASS',next_step='单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。'))
        self.wait(self.job/'worker/ready.json',715)
        ready=json.loads((self.job/'worker/ready.json').read_text())
        if ready['CPU_probe'] or ready['pid']!=self.worker.pid:raise ValueError('actual GPU child readiness identity')
        admission=json.loads((self.job/'worker/INTERFACE-ADMISSION.json').read_text())
        if admission['status']!='PASS' or admission['module']!='modules.model' or admission['layers']!=30:raise ValueError('production GPU admission missing')
        self.event('ADMISSION',dict(status='PASS',admission=admission,actual_gpu_worker_pid=self.worker.pid,
            next_step='真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。'))
        kernel=self.command('kernel','kernel')
        self.passed.append('KERNEL_DIAGNOSTIC');self.event('KERNEL',dict(status='PASS_KERNEL_DIAGNOSTIC',result=kernel,next_step='kernel数学/隔离关通过（非target验收）；恢复真实fixture，验证29变化query全30层。'))
        captured=self.command('capture','capture');probe=self.command('resource_probe','resource_probe')
        self.passed.append('CAPTURE');self.event('CAPTURE',dict(status='PASS',capture=captured,probe=probe,next_step='进入冻结HC与S全30层数值、十二类失效及13行隔离；严格门不降低。'))
        hc=self.command('equivalence_HC','sequence',arm='HC',timed=False)
        shared=self.command('equivalence_S','sequence',arm='S',timed=False)
        for i in range(1,13):self.command(f'invalidation_{i:02d}','invalidation',case=i)
        for a in ['R','HC','S']:self.command('isolation_'+a,'isolation',arm=a)
        strict=shared['bitwise_all_outputs'] and all(json.loads((self.job/'results'/f'invalidation_{i:02d}.json').read_text())['strict_all_S_numerical_checks'] for i in range(1,13))
        self.passed.append('VALIDATION' if strict else 'DIAGNOSTIC_VALIDATION_ONLY')
        self.event('VALIDATION',dict(status='PASS' if strict else 'DIAGNOSTIC_ONLY_NOT_ACCEPTED',strict_S_bitwise=strict,
            HC_bitwise=hc['bitwise_all_outputs'],S_numerical=shared['numerical'],invalidation_classes=12,
            other13_isolated=True,next_step='开始R/S六交错块OFF；若非逐位，仅使用用户授权诊断分支，不投入真实策略。'))
        for a in ['R','S']:self.command('warmup_'+a,'sequence',arm=a,timed=False)
        rows=[]
        for block,order in enumerate(self.p['main_orders']):
            for a in order:
                row=self.command(f'OFF_{block}_{a}','sequence',arm=a,timed=True);row['block']=block;rows.append(row)
        from prefix_compare import off_analysis
        off=off_analysis(rows,diagnostic_allowed=self.p['diagnostic_nonbitwise_approved']);write(self.job/'OFF-RESULT.json',off,exclusive=True)
        if off['status'] not in ('PASS','DIAGNOSTIC_ONLY_NOT_ACCEPTED'):raise RuntimeError('OFF stability gate failed; no extra samples')
        self.passed.append('OFF' if strict else 'DIAGNOSTIC_OFF_ONLY')
        self.event('OFF',{**off,'next_step':'GPU性能样本已保留；释放缓存、关闭写前计数、清理本任务进程，不自动训练或闭环。'})
        finish=self.command('finish','finish')
        if self.worker.wait(timeout=30)!=0:raise RuntimeError('worker final exit')
        from prefix_compare import aggregate_strict
        strict=aggregate_strict(strict,off,finish)
        cleanup=self.cleanup_audit();freeze(self.p)
        if self.counter_recovery['status']!='CLOSED' or self.counter_recovery['ledger']['target_inflight']:raise RuntimeError('closed journal required')
        return dict(status='PASS' if strict else 'DIAGNOSTIC_ONLY_NOT_ACCEPTED',OFF=off,kernel=kernel,
            numerical=shared['numerical'],finish=finish,budget=self.budget_summary(),cleanup=cleanup,counter_recovery=self.counter_recovery,
            GPU_elapsed_sec=self.gpu_end-self.gpu_start,aggregate_GPU_sec=self.gpu_end-self.gpu_start+self.p['aggregate_prior_GPU_seconds'],
            aggregate_target_intents=self.budget_summary()['target_attempted']+183,passed_gates=self.passed,
            next_step='先验收共享KV数值与单worker完整verifier速度/显存；非逐位则不能投用，下一步制定数值准入/selector决策复核。通过后另规划完整policy与真实闭环，不自动训练或mega Transformer。')

    def brief(self,result):
        (self.job/'BRIEF.md').write_text('# 投机采样：CFG2七候选零拷贝共享KV\n\n'+stamp()+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n下一步：'+result['next_step']+'\n')

    def run(self,mode):
        with (self.job/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write(self.job/('CPU-CONTROLLER.json' if mode in ('cpu','auto') else 'GPU-CONTROLLER.json'),dict(pid=os.getpid(),timestamp=stamp()),exclusive=True)
            try:
                result=self.execute_phases(mode)
                if mode=='cpu':return
                write(self.job/'FINAL_RESULT.json',result,exclusive=True)
                self.stage='COMPLETED';self.brief(result);self.event('FINAL',result)
            except Exception as exc:
                self.stop.set();self.stage='FAILED'
                result=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),timed_out=self.timed_out,
                    budget=self.budget_summary(),passed_gates=self.passed,
                    next_step='失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。')
                try:result['cleanup']=self.cleanup_audit()
                except Exception as cleanup:result['cleanup_error']=repr(cleanup)
                result['budget']=self.budget_summary();result['counter_recovery']=self.counter_recovery
                write(self.job/'FAILED.json',result,exclusive=True);self.brief(result)
                try:self.event('FAILED',result)
                except Exception:(self.job/'HW_WRITE_FAILED.txt').write_text(traceback.format_exc())
                raise
            finally:
                self.done.set();self.cleanup()
                if self.nvml is not None:self.nvml.close()

    def execute_phases(self,mode):
        # Detached handoff, not a CPU-gate bypass. gpu() still independently
        # validates the fresh plan-bound CPU gate before resource/model entry.
        if mode in ('cpu','auto'):self.cpu()
        if mode=='cpu':return None
        if mode=='auto':
            write(self.job/'GPU-CONTROLLER.json',dict(pid=os.getpid(),timestamp=stamp(),
                  after_fresh_CPU_preflight=True),exclusive=True)
        return self.gpu()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--mode',choices=['cpu','gpu','auto'],required=True)
    a=ap.parse_args();controller=Supervisor(a.job)
    signal.signal(signal.SIGTERM,lambda *_:controller.expire('supervisor termination signal'))
    controller.run(a.mode)


if __name__=='__main__':main()
