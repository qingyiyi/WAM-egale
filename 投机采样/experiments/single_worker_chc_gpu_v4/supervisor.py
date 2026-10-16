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
        if self.gpu_start is not None and self.gpu_end is None and time.monotonic()>=self.gpu_start+2700-15:
            self.expire('45min deadline with15s cleanup reserve');raise TimeoutError('GPU safety deadline')
        if shutil.disk_usage(self.job).free<20*2**30:raise RuntimeError('disk below20GiB')

    def environment(self,*,cpu=False):
        e={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_','NSYS_'))}
        e.pop('CUDA_LAUNCH_BLOCKING',None);ref=self.p['frozen_reference']
        e.update(dict(CUDA_VISIBLE_DEVICES='' if cpu else '0',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',
            OMP_NUM_THREADS='7',MKL_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',NUMEXPR_NUM_THREADS='7',
            P0_CHC_V4_ENABLED='1',P0_CHC_V4_GPU_APPROVED='1',
            LINGBOT_SEED='20260922',LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',
            LINGBOT_VA_LIBERO_SUITE='libero_10',LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'],
            WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',WAM_P0_ACTION_DRAFT_MODE='off',
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=self.p['frozen_A_off'],WAM_P0_ACTION_DRAFT_SELECTOR='global',
            WAM_P0_ACTION_DRAFT_THRESHOLD=str(self.p['threshold']),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(self.p['ports'][0]),
            RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PYTHONPATH=str(OWN)+':'+self.p['worktree']+':'+str(ROOT/'投机采样/scripts')))
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
                self.gpu_start=time.monotonic();self.ledger.deadline=self.gpu_start+2700-15
                write(self.job/'GPU-ATTEMPT.json',dict(timestamp=stamp(),attempt=1,
                    job=str(self.job),execution_revision=REVISION,
                    monotonic_start=self.gpu_start,monotonic_deadline=self.gpu_start+2700,
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
                    if time.monotonic()>=self.gpu_start+2700-15:self.expire('45min independent watchdog deadline');return
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
        tasks=[('test_job_cpu.py',['--job',str(self.job),'--out',str(self.job/'preflight/job-tests.json')],'job-tests.log'),
               ('test_counter_cpu.py',['--job',str(self.job),'--out',str(self.job/'preflight/counter-tests.json')],'counter-tests.log'),
               ('test_loader_cpu.py',['--out',str(self.job/'preflight/loader-tests.json')],'loader-tests.log'),
               ('test_cpu.py',['--out',str(self.job/'preflight/interface-tests.json')],'interface-tests.log'),
               ('worker.py',['--job',str(self.job),'--cpu-probe'],'worker-probe.log')]
        for script,args,log in tasks:
            child=self.spawn([self.p['model_python'],str(OWN/script),*args],self.job/'preflight'/log,cpu=True)
            if child.wait(timeout=900)!=0:raise RuntimeError('new-job CPU preflight failed: '+script)
        j=json.loads((self.job/'preflight/job-tests.json').read_text());t=json.loads((self.job/'preflight/interface-tests.json').read_text())
        l=json.loads((self.job/'preflight/loader-tests.json').read_text())
        c=json.loads((self.job/'preflight/counter-tests.json').read_text())
        probe=json.loads((self.job/'preflight/worker-CPU-probe.json').read_text())
        if j['status']!='PASS' or c['status']!='PASS' or l['status']!='PASS' or t['status']!='PASS' or probe['status']!='PASS':raise RuntimeError('CPU result status')
        if c['CUDA_initialized'] or c['forbidden_CUDA_attempts']:raise RuntimeError('CPU counter initialized CUDA')
        if l['CUDA_initialized'] or t['CUDA_initialized'] or probe['CUDA_initialized']:raise RuntimeError('CPU initialized CUDA')
        if probe['production_model_module']!='modules.model':raise RuntimeError('real production loader probe missing')
        pc=probe['recovered_counter']
        if pc['status']!='CLOSED' or pc['ledger']['target_attempted']!=87 or pc['ledger']['target_completed']!=87:
            raise RuntimeError('same worker CPU counter probe missing')
        if (self.job/'worker/ready.json').exists() or (self.job/'GPU-ATTEMPT.json').exists():raise RuntimeError('CPU probe polluted GPU readiness/attempt')
        freeze(self.p)
        g=dict(status='PASS',plan_sha256=digest(self.job/'plan.json'),implementation_sha256=self.p['implementation_sha256'],
            job_tests=j['tests_run'],real_interface_tests=t['tests_run'],same_actual_worker_CPU_probe=True,
            production_loader_positive_negative_tests=l['tests_run'],real_production_loader_CPU_probe=True,
            counter_persistence_and_actual_worker_termination_tests=c['tests_run'],
            all30_FP32_BF16_counter_and_state_equivalence=True,
            CUDA_initialized=False,new_GPU_started=False,new_job_path_isolation=True)
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
        self.wait(out,900 if op in ('capture','diagnostic') else 300)
        row=json.loads(out.read_text());actual=row.get('ledger',{})
        recovery=self.read_counter()
        if recovery.get('ledger')!=actual:raise ValueError('worker ack differs from persisted owned counter')
        self.ledger.attempted=actual.get('target_attempted',self.ledger.attempted)
        self.ledger.completed=actual.get('target_completed',self.ledger.completed)
        self.ledger.stage_attempted=actual.get('stage_attempted',0)
        self.status()
        if row.get('status')!='PASS':raise RuntimeError('worker gate failed: '+row.get('error','unknown'))
        if actual['target_reserved']!=self.ledger.reserved or actual['target_attempted']>900:raise ValueError('worker/controller budget mismatch')
        write(self.job/'results'/(stage+'.json'),row,exclusive=True);return row

    def gpu(self):
        self.p=enabled_plan(self.job,cpu_gate=True);self.passed=['CPU'];self.resource()
        self.stage='GPU_LOADING'
        self.spawn([self.p['model_python'],str(OWN/'worker.py'),'--job',str(self.job)],self.job/'worker/model.log',gpu=True)
        self.event('START',dict(status='PASS',next_step='单worker开始计时（含加载和清理）；先恢复原fixture并捕获29变化输入，0 EGL/完整chunk。'))
        self.wait(self.job/'worker/ready.json',900)
        ready=json.loads((self.job/'worker/ready.json').read_text())
        if ready['CPU_probe'] or ready['pid']!=self.worker.pid:raise ValueError('actual GPU child readiness identity')
        admission=json.loads((self.job/'worker/INTERFACE-ADMISSION.json').read_text())
        if admission['status']!='PASS' or admission['module']!='modules.model' or admission['layers']!=30:raise ValueError('production GPU admission missing')
        self.event('ADMISSION',dict(status='PASS',admission=admission,actual_gpu_worker_pid=self.worker.pid,
            next_step='真实GPU生产类全30层接口准入通过，0 target前向；现在恢复/捕获原始变化query。'))
        captured=self.command('capture','capture');probe=self.command('resource_probe','resource_probe')
        self.passed.append('CAPTURE');self.event('CAPTURE',dict(status='PASS',capture=captured,probe=probe,next_step='进入C/HC全30层29变化输出、十二类失效与13行隔离关；未通过不测速。'))
        for a in ['C','HC']:self.command('equivalence_'+a,'sequence',arm=a,timed=False)
        for i in range(1,13):self.command(f'invalidation_{i:02d}','invalidation',case=i)
        for a in ['R','C','HC']:self.command('isolation_'+a,'isolation',arm=a)
        self.passed.append('VALIDATION');self.event('VALIDATION',dict(status='PASS',all30_changing29_bitwise=True,invalidation_classes=12,
            other13_isolated=True,next_step='数值/失效/显存关通过；开始三臂warmup与六平衡顺序OFF，冷构建/staging/键检查/释放收费。'))
        for a in ['R','C','HC']:self.command('warmup_'+a,'sequence',arm=a,timed=False)
        rows=[]
        for block,order in enumerate(self.p['main_orders']):
            for a in order:
                row=self.command(f'OFF_{block}_{a}','sequence',arm=a,timed=True);row['block']=block;rows.append(row)
        from gpu_compare import off_analysis,trace_analysis
        off=off_analysis(rows);write(self.job/'OFF-RESULT.json',off,exclusive=True)
        if off['status']!='PASS':raise RuntimeError('OFF stability gate MAD/median above7%; no repetitions added')
        self.passed.append('OFF');self.event('OFF',{**off,'next_step':'OFF稳定性关通过，保留正/负/混合结果；同进程现在才开R/HC独立ON诊断，不以ON时间作性能主测。'})
        diag=self.command('diagnostic','diagnostic');finish=self.command('finish','finish')
        if self.worker.wait(timeout=30)!=0:raise RuntimeError('worker final exit')
        cleanup=self.cleanup_audit();freeze(self.p)
        if self.counter_recovery['status']!='CLOSED' or self.counter_recovery['ledger']['target_inflight']:
            raise RuntimeError('normal GPU finish requires closed persisted counter')
        self.stage='CPU_DIAGNOSTIC_ANALYSIS';self.status()
        trace=trace_analysis(diag['trace']);self.passed.append('DIAGNOSTIC')
        self.event('DIAGNOSTIC',dict(status='PASS',trace_summary=trace,finish=finish,cleanup=cleanup,next_step='GPU已清理；CPU汇总完整verifier成本、全层显存和trace，不自动下一实验。'))
        return dict(status='PASS',OFF=off,diagnostic=trace,budget=self.budget_summary(),cleanup=cleanup,
            counter_recovery=self.counter_recovery,
            GPU_elapsed_sec=self.gpu_end-self.gpu_start,passed_gates=self.passed,
            next_step='先验收本轮全verifier速度/显存与负结果；如有稳定净收益，再单独规划完整单worker policy/冻结target闭环。若无收益，先CPU归因cat/attention、host键开销与显存，不自动mega、扩大GPU或训练。')

    def brief(self,result):
        lines=['# 投机采样：单worker全层C/HC验证简报','',stamp()+'（Asia/Shanghai）','',
            f"- 状态：**{result['status']}**；通过关：{', '.join(self.passed) or '尚无'}。",
            f"- GPU启动：{self.gpu_start is not None}；target预算/计数：{json.dumps(result.get('budget',self.budget_summary()),ensure_ascii=False)}，硬限900；0episode/EGL/完整chunk/训练。",
            '- 原batch14/原SDPA/公共模型不变；C/HC仍14行展开复用，非新prefix kernel，非TILE。']
        if result.get('error'):lines.append('- 停止原因：'+result['error'])
        if self.gpu_start is not None:lines.append(f'- GPU加载至清理：{((self.gpu_end or time.monotonic())-self.gpu_start)/60:.2f}分钟（45分钟安全限时）。')
        if (self.job/'OFF-RESULT.json').exists():
            off=json.loads((self.job/'OFF-RESULT.json').read_text())
            lines+=['','## 完整verifier OFF（29变化查询、30层，不是完整policy）','','|臂|中位秒|MAD/median|额外peak GiB|','|---|---:|---:|---:|']
            for a,v in off['metrics'].items():lines.append(f"|{a}|{v['median_sec']:.4f}|{v['MAD_over_median']:.4%}|{v['max_extra_peak_bytes']/2**30:.3f}|")
            for v in off['effects']:lines.append(f"- {v['comparison']}：{v['improvement_percent']:.2f}%降低；{v['signal']}；六块配对比值 {v['paired_block_ratios']}。")
        lines+=['','## 进展边界与下一步','','这是同固定状态的完整verifier replay；不证明完整policy、成功率、环境步数或E2E。失败及所有原始样本保留，不补跑。',result['next_step'],
            '',f"[状态]({self.job/'status.json'})；[CPU预检]({self.job/'preflight/CPU-GATE.json'})；[合同]({self.job/'plan.json'})。"]
        (self.job/'BRIEF.md').write_text('\n'.join(lines)+'\n')

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
