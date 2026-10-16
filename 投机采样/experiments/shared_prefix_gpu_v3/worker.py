"""One-shot GPU entry integrating the CPU-validated write-ahead counters.

Public math, batch, attention and all C/HC methods are preserved. Fresh
v4 flags, plan, CPU gate and owned GPU claim precede model/CUDA imports.
"""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent.parent/'single_worker_chc_gpu_v4'
import sys
if str(BASE) not in sys.path:sys.path.append(str(BASE))

from task_contract import OWN, read_cpu_plan, gpu_entry_gate, counter_identity, Ledger, STAGES, write, stamp, digest
from counter_journal import PersistentLedger, recover


class WorkerStop(RuntimeError):
    pass


def load_snapshot():
    path=BASE.parent/'fixed_load_cost/snapshot.py'
    spec=importlib.util.spec_from_file_location('single_chc_original_snapshot',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Runtime:
    def __init__(self,server,source,job,plan,*,CPU_probe=False):
        if not CPU_probe:
            authorized=gpu_entry_gate(job)
            if plan!=authorized:raise ValueError('GPU Runtime plan identity')
        import torch
        from prefix_adapter import SharedPrefix
        if CPU_probe and server.device.type != 'cpu':
            raise ValueError('CPU probe device required')
        self.t=torch;self.s=server;self.m=server.transformer;self.source=source
        self.job=Path(job);self.p=plan;self.cpu=CPU_probe;self.device=server.device
        self.base=self.job/('preflight/cpu-worker' if CPU_probe else 'worker')
        self.snapshot=load_snapshot();self.ledger=Ledger()
        self.strict_all_S_checks=True;self.control=None;self.capture_data=None;self.hooks=[];self._stop_signal=0
        if not CPU_probe:
            if (server._cache_mode!='original' or len(self.m.blocks)!=30 or server._phase_profile_enabled
                or server._p0_trace is not None or server._p0_state_teacher is not None
                or server._nsys_profile_mode!='off' or server.job_config.action_num_inference_steps!=50
                or server.job_config.num_inference_steps!=20 or server.job_config.action_guidance_scale!=1
                or str(next(server.text_encoder.parameters()).device)!='cpu'
                or self.device.type!='cuda'):raise ValueError('frozen model config')
            start=json.loads((self.job/'GPU-ATTEMPT.json').read_text())
            self.ledger.deadline=start['monotonic_deadline']-15
        self.admission=self.check_interface()
        self.counter_identity=None
        if plan.get('counter_persistence_enabled'):
            if CPU_probe and plan!=read_cpu_plan(self.job):raise ValueError('CPU counter plan binding')
            self.counter_identity=counter_identity(self.job,CPU=CPU_probe)
            deadline=self.ledger.deadline
            self.ledger=PersistentLedger(self.base/'counter.jsonl',self.counter_identity,enabled=True,
                                         on_commit=self.cpu_counter_boundary if CPU_probe else None)
            self.ledger.deadline=deadline
        write(self.base/'INTERFACE-ADMISSION.json',self.admission)
        original=self.m.forward
        def counted(input_dict,update_cache=0,cache_name='pos',action_mode=False,train_mode=False,candidate_cache_rows=None):
            if train_mode:raise RuntimeError('training forbidden')
            self.check_stop();self.ledger.before_forward()
            try:out=original(input_dict,update_cache,cache_name,action_mode,train_mode,candidate_cache_rows)
            except BaseException as exc:
                if isinstance(self.ledger,PersistentLedger):self.ledger.forward_failed(exc)
                raise
            if self.cpu and self.p.get('CPU_test_fault')=='after_model_return_2' and self.ledger.attempted==2:
                self.cpu_pause('after_model_return_2')
            self.ledger.after_forward();return out
        # Explicit real signature is preserved for adapter.bind() and capture.
        self.m.forward=counted
        self.saved=[]
        if source is not None:source.save_async=lambda value,path:self.saved.append(Path(path).name)
        write(self.base/'ready.json',dict(status='PASS',pid=os.getpid(),device=str(self.device),
            CPU_probe=CPU_probe,training=False,layers=len(self.m.blocks),public_source_modified=False,
            CUDA_initialized=self.t.cuda.is_initialized()))

    def check_stop(self):
        if self._stop_signal:raise WorkerStop('received '+signal.Signals(self._stop_signal).name)
        if (self.job/'STOP_REQUESTED').exists():raise RuntimeError('owned stop requested')
        if self.ledger.deadline is not None and time.monotonic()>=self.ledger.deadline:raise TimeoutError('GPU cleanup-reserved deadline')

    def cpu_pause(self,name):
        if not self.cpu or self.p.get('CPU_only') is not True:raise ValueError('fault barrier CPU-only')
        write(self.base/'PAUSE.json',dict(boundary=name,ledger=self.ledger.summary(),pid=os.getpid()),exclusive=True)
        deadline=time.monotonic()+30
        while not (self.base/'RELEASE').exists():
            if time.monotonic()>deadline:raise TimeoutError('CPU fault barrier timeout')
            time.sleep(.01)

    def cpu_counter_boundary(self,event,state):
        mode=self.p.get('CPU_test_fault')
        if (mode=='after_BEGIN' and event=='BEGIN' or
                mode=='after_BEFORE_2' and event=='BEFORE' and state['target_attempted']==2):
            self.cpu_pause(mode)

    @contextmanager
    def termination_handler(self):
        # The handler only sets a flag: no file I/O or reentrant journal write.
        old=signal.getsignal(signal.SIGTERM)
        def stop(sig,_frame):self._stop_signal=sig
        signal.signal(signal.SIGTERM,stop)
        try:yield
        finally:signal.signal(signal.SIGTERM,old)

    def close_counter(self,reason):
        if isinstance(self.ledger,PersistentLedger):self.ledger.close(reason)

    def check_interface(self):
        from prefix_adapter import SharedPrefix
        return SharedPrefix(self.m,enabled=True,allow_cuda=not self.cpu,cpu_reference_enabled=self.cpu).admit()

    def sync(self):
        if not self.cpu:self.t.cuda.synchronize(self.device)

    def memory(self):
        if self.cpu:return dict(allocated=0,peak=0,free=0)
        return dict(allocated=self.t.cuda.memory_allocated(self.device),
                    peak=self.t.cuda.max_memory_allocated(self.device),free=self.t.cuda.mem_get_info(self.device)[0])

    def restore_history(self):
        def restore():
            for b,c in zip(self.m.blocks,self.history):self.snapshot.restore_cache(b.attn1.attn_caches[self.s.cache_name],c)
        if self.control is None:restore()
        else:self.control.external_call(restore,reason='fixed first-candidate history restore')
        self.snapshot_state_check()

    def snapshot_state_check(self):
        # All CPU copies/finite checks/comparison occur OUTSIDE the OFF timer.
        for i,(b,old) in enumerate(zip(self.m.blocks,self.history)):
            now=self.snapshot.capture_cache(b.attn1.attn_caches[self.s.cache_name])
            if now.keys()!=old.keys() or not all(self.t.equal(now[k],old[k]) if isinstance(now[k],self.t.Tensor) else now[k]==old[k] for k in now):
                raise ValueError('all-layer valid history/metadata drift at '+str(i))

    def capture(self):
        from capture import capture_queries
        t=self.t;s=self.s
        if self.capture_data is not None:raise ValueError('capture retry')
        # Recheck before torch.load, reset/preload, infer or any target math.
        # Keep capture ORIGINAL; this check does not install an adapter.
        self.admission=self.check_interface()
        payload=t.load(self.p['snapshot_reference']['path'],map_location='cpu',weights_only=False)
        if (self.snapshot.fingerprint(payload['state'])!=self.p['snapshot_reference']['state_hash']
                or self.snapshot.fingerprint(payload['request'])!=self.p['snapshot_reference']['request_hash']):raise ValueError('actual fixture contents mismatch')
        # Geometry must come from actual reset. Do not initialize versionless
        # pools under inference_mode; the public reset itself is no_grad.
        s.infer(dict(reset=True,prompt=None))
        previous=s._p0_action_draft_mode
        try:
            s._p0_action_draft_mode='verifier_k2_b2';pred=s._load_p0_action_draft_b2_predictor()
            pred.model.to(s.device).eval()
        finally:s._p0_action_draft_mode=previous
        if pred.model.output_dim!=112:raise ValueError('A_off geometry')
        self.snapshot.restore(s,payload['state']);restored=self.snapshot.capture(s)
        if self.snapshot.fingerprint(restored)!=payload['state_hash']:raise ValueError('real all-block/RNG restore mismatch')
        del restored
        s._p0_action_draft_mode='verifier_k2_b2'
        # capture_queries has its own cap52, stops before the terminal math;
        # counted() independently enforces the stage/global target budget.
        self.r_attention={};saved_backends=[]
        try:
            for layer,block in enumerate(self.m.blocks):
                a=block.attn1;original_op=a.attn_op;saved_backends.append((a,original_op))
                def record(q,k,v,_layer=layer,_original=original_op):
                    out=_original(q,k,v)
                    if q.shape[0]==14 and q.shape[1]==16 and _layer not in self.r_attention:
                        self.r_attention[_layer]=dict(q=q.detach().cpu().clone(),
                            k=k[:,-16:].detach().cpu().clone(),v=v[:,-16:].detach().cpu().clone(),
                            out=out.detach().cpu().clone())
                    return out
                a.attn_op=record
            with t.no_grad():self.capture_data=capture_queries(s,payload['request'],enabled=True,allow_cuda=True)
        finally:
            for a,original_op in saved_backends:a.attn_op=original_op
        if len(self.r_attention)!=30:raise ValueError('first original candidate attention capture missing')
        self.queries=self.capture_data['queries'];self.history=self.capture_data['history_before_candidates']
        if len(self.queries)!=29 or len(self.history)!=30 or any(c['slots'].numel()!=2144 for c in self.history):raise ValueError('actual full30/29/long-history capture')
        for q in self.queries:
            if q.input_dict['noisy_latents'].shape!=(14,30,4,4,1) or q.input_dict['noisy_latents'].dtype!=t.bfloat16:raise ValueError('actual batch/BF16 geometry')
        for b in self.m.blocks:
            if b.attn1.heads!=24 or b.attn1.inner_dim!=3072:raise ValueError('actual heads geometry')
        del payload;gc.collect()
        self.restore_history();self.install_control()
        self.initial_rng=self.snapshot.fingerprint(self.snapshot.rng(True))
        return dict(status='PASS',calls=self.capture_data['calls'],captured_queries=29,
            layers=30,history_tokens=2144,actual_PT_contents_verified=True,
            history_constant_across_all29=True,stopped_before_action_terminal=True,
            original_outputs_hash=[self.snapshot.fingerprint(q.output) for q in self.queries],
            complete_policy_chunks=0,EGL=0)

    def install_control(self):
        from prefix_adapter import SharedPrefix
        self.control=SharedPrefix(self.m,enabled=True,allow_cuda=not self.cpu,cpu_reference_enabled=self.cpu).install()
        # The public pools/parameters MUST have usable mutation counters.
        with self.control.lease(request='admission',arm='S',device=self.device):
            for b in self.m.blocks:
                for v in b.attn1.attn_caches[self.s.cache_name].values():_ = v._version

    def call(self,q,rows):return q.replay(self.m,rows=rows,device=self.device,allow_cuda=not self.cpu)

    def sequence(self,arm,*,timed=False,diagnostic=False,subset=None,request='fixed'):
        t=self.t;c=self.control;qs=self.queries if subset is None else subset
        rng=self.snapshot.fingerprint(self.snapshot.rng(not self.cpu))
        self.restore_history();c.diagnostic=diagnostic
        self.sync();baseline=self.memory()['allocated']
        if not self.cpu:t.cuda.reset_peak_memory_stats(self.device)
        events=None
        if timed and not self.cpu:events=(t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True))
        if arm=='S' and request=='equivalence_S':c.record_first=True;c.first_attention={}
        outputs=[];scope=t.profiler.record_function('CHC.VERIFIER.'+arm) if diagnostic else nullcontext()
        begin=time.monotonic_ns()
        if events:events[0].record()
        with scope,t.inference_mode():
            with c.lease(request=request,arm=arm,device=self.device) as token:
                for q in qs:outputs.append(self.call(q,token.rows))
                if arm!='R' and len(c.entries)!=30:raise ValueError('incomplete all30 cache')
        if events:events[1].record()
        self.sync();end=time.monotonic_ns();mem=self.memory()
        if c.entries or c.payload_bytes:raise ValueError('lease leaked history references/cache')
        if not self.cpu and (mem['peak']-baseline>12*2**30 or mem['free']<8*2**30):raise ValueError('12GiB extra peak/8GiB headroom gate')
        # Numerical checking, hashes and state comparison are not timed.
        from prefix_compare import numerical
        cpu_outputs=[out.detach().cpu() for out in outputs]
        numeric=[numerical(out,q.output) for out,q in zip(cpu_outputs,qs)]
        bitwise=all(v['bitwise'] for v in numeric)
        if not self.cpu and arm=='S' and request=='equivalence_S':
            write(self.job/'numerical/S-all29.json',dict(numerical=numeric,strict_all=bitwise),exclusive=True)
            t.save(dict(original=[q.output for q in qs],shared=cpu_outputs),self.job/'numerical/S-all29-outputs.pt')
            self.attention_attribution()
            c.record_first=False;c.first_attention={}
        for i,(out,q) in enumerate(zip(cpu_outputs,qs)):self.check_numerical(out,q.output,arm,request+'-'+str(i))
        self.snapshot_state_check()
        if self.snapshot.fingerprint(self.snapshot.rng(not self.cpu))!=rng:raise ValueError('RNG changed')
        if diagnostic and arm=='HC':
            expected=dict(metadata=1,gather=1,row_check=1,expand=1,cat=len(qs),SDPA=len(qs))
            if len(c.ops)!=30 or any(v!=expected for v in c.ops.values()):raise ValueError('all30 HC ON counters')
        return dict(status='PASS',arm=arm,wall_sec=(end-begin)/1e9,
            CUDA_event_ms=events[0].elapsed_time(events[1]) if events else None,
            CUDA_event_is_span_not_kernel_busy=True,cold_setup_staging_keys_release_included=True,
            profile_OFF=not diagnostic,query_count=len(qs),bitwise_all_outputs=bitwise,numerical=numeric,
            all30_history_metadata_RNG_unchanged=True,baseline_allocated_bytes=baseline,
            peak_allocated_bytes=mem['peak'],extra_peak_bytes=mem['peak']-baseline,
            free_GPU_bytes=mem['free'],operations=c.ops if diagnostic else None,shared_prefix_index_peak_bytes=c.prefix_peak_bytes if arm=='S' else None,
            original_CFG2_pool_zero_copy=(arm=='S'),history14_allocated=(arm=='HC'),fullKV14_materialized=(arm!='S'))

    def attention_attribution(self):
        from prefix_attention import shared_attention
        from prefix_compare import numerical
        from prefix_micro import record_intent
        if self.cpu or len(self.r_attention)!=30 or len(self.control.first_attention)!=30:
            raise ValueError('actual all30 attention attribution capture required')
        rows=self.t.tensor([0]*7+[1]*7,dtype=self.t.long,device=self.device)
        result=[]
        for layer,block in enumerate(self.m.blocks):
            self.check_stop();r=self.r_attention[layer];s=self.control.first_attention[layer]
            cache=block.attn1.attn_caches[self.s.cache_name];slots=cache['mask'].nonzero().flatten()
            q,k,v=(r[name].to(self.device) for name in ('q','k','v'))
            record_intent(self,'same-actual-R-QKV-layer-'+str(layer))
            out=shared_attention(q,cache['k'],cache['v'],k,v,slots,rows,enabled=True,allow_cuda=True,maps_validated=True)
            self.sync()
            result.append(dict(layer=layer,same_R_QKV_new_kernel_vs_R_attention=numerical(out.cpu(),r['out']),
                propagated_S_Q_vs_R_Q=numerical(s['q'],r['q']),
                propagated_S_currentK_vs_R_currentK=numerical(s['k'],r['k']),
                S_attention_vs_R_attention_with_different_later_inputs=numerical(s['out'],r['out'])))
        write(self.job/'numerical/ATTENTION-ATTRIBUTION.json',dict(scope='all30 actual first candidate; identical R QKV isolated from propagation',
            rows=result,extra_target_forwards=0,direct_micro_intents=getattr(self,'direct_micro_count',0),
            mathematical_vs_bitwise_distinguished=True),exclusive=True)
        self.r_attention={}

    def check_numerical(self,actual,gold,arm,name):
        from prefix_compare import numerical
        row=numerical(actual,gold)
        if not self.cpu and arm=='S':self.strict_all_S_checks &= row['bitwise']
        if self.cpu:
            self.t.testing.assert_close(actual,gold,rtol=2e-5,atol=5e-6)
        elif not row['bitwise']:
            write(self.job/'numerical'/(name+'.json'),row,exclusive=True)
            if arm!='S' or not self.p['diagnostic_nonbitwise_approved']:
                raise ValueError('strict bitwise gate failed for '+arm+'; numerical report preserved; no OFF or silent tolerance')
            if not row['finite'] or row['relative_RMS_error']>.01:
                raise ValueError('S diagnostic corruption ceiling failed; not a formal acceptance tolerance')
        return row

    def probe(self):
        row=self.sequence('R',subset=self.queries[:1],request='R_resource_probe')
        if not self.cpu:
            self.t.cuda.empty_cache();self.sync();row['free_after_R_probe']=self.memory()['free']
            if row['free_after_R_probe']<20*2**30:raise ValueError('before-cache free GPU below20GiB')
        return row

    def invalidation(self,case):
        from adapter import LeaseError
        t=self.t;c=self.control;q=self.queries[0];self.restore_history()
        def ordinary(update):
            inp={k:v[[0,7]].to(self.device) if isinstance(v,t.Tensor) and v.shape[0]==14 else v.to(self.device) if isinstance(v,t.Tensor) else v for k,v in q.input_dict.items()}
            return self.m(inp,update_cache=update,cache_name=self.s.cache_name,action_mode=True)
        def mutate(token):
            if case==1:ordinary(1)
            elif case==2:ordinary(2)
            elif case==3:self.m.clear_pred_cache(self.s.cache_name)
            elif case==4:
                self.m.clear_cache(self.s.cache_name)
                for b,saved in zip(self.m.blocks,self.history):
                    sh=saved['pool_shape'];b.attn1.init_kv_cache(self.s.cache_name,sh[1],sh[2],sh[3],self.device,b.attn1.to_q.weight.dtype,sh[0])
                self.restore_history()
            elif case==5:self.restore_history()
            elif case==6:
                b=self.m.blocks[-1].attn1.attn_caches[self.s.cache_name];slot=int(self.history[-1]['slots'][0]);b['v'][:,slot].add_(.125)
            elif case==7:
                b=self.m.blocks[-1].attn1.attn_caches[self.s.cache_name]
                valid=b['mask'].nonzero().flatten();invalid=(~b['mask']).nonzero().flatten()
                if not len(invalid):raise ValueError('no unused slot for replacement')
                a,z=valid[0],invalid[0];b['k'][:,z]=b['k'][:,a];b['v'][:,z]=b['v'][:,a];b['id'][z]=b['id'][a]
                b['mask'][a]=False;b['mask'][z]=True
            elif case==8:
                with c.mutation('controlled full-capacity fill before real eviction'):
                    for block in self.m.blocks:
                        b=block.attn1.attn_caches[self.s.cache_name];invalid=~b['mask']
                        b['k'][:,invalid]=0;b['v'][:,invalid]=0;b['mask'].fill_(True)
                        b['id'].copy_(t.arange(b['mask'].numel(),device=self.device))
                ordinary(0)
            elif case==9:ordinary(0)
            elif case==10:token.rows[0]=1
            elif case==11:c.invalidate('new request/cache_name generation')
            elif case==12:c.model_generation+=1  # host generation only; frozen weights NEVER changed
            else:raise ValueError('unknown invalidation')
        # Exercise both C and HC, rebuilding only after the old lease ends.
        for arm in ['S']:
            self.restore_history()
            with t.no_grad(),c.lease(request='before-'+str(case)+arm,arm=arm,device=self.device) as token:
                self.call(q,token.rows);mutate(token)
                try:self.call(q,token.rows)
                except LeaseError:pass
                else:raise ValueError('stale lease accepted')
                if c.entries:raise ValueError('stale cache retained')
            rows=t.tensor([0]*7+[1]*7,dtype=t.long,device=self.device)
            with t.inference_mode():gold=self.call(q,rows)
            with t.inference_mode(),c.lease(request='new-'+str(case)+arm,arm=arm,device=self.device) as token:
                rebuilt=self.call(q,token.rows)
            self.check_numerical(rebuilt.detach().cpu(),gold.detach().cpu(),'S','invalidation-'+str(case))
        self.restore_history()
        return dict(status='PASS',case=case,S_stale_refused=True,new_history_R_compared=True,
                    frozen_model_weights_not_modified=True)

    def isolation(self,arm):
        from capture import Query
        self.restore_history();q=self.queries[0]
        changed={k:v.clone() if isinstance(v,self.t.Tensor) else v for k,v in q.input_dict.items()}
        changed['noisy_latents'][0].add_(.25)
        other=Query(changed,q.kwargs,q.output)
        with self.t.inference_mode(),self.control.lease(request='isolation-'+arm,arm=arm,device=self.device) as token:
            a=self.call(q,token.rows);b=self.call(other,token.rows)
        if not self.t.equal(a[1:],b[1:]) or self.t.equal(a[0],b[0]):raise ValueError('13-row isolation/nondegenerate perturbation gate')
        self.snapshot_state_check();return dict(status='PASS',arm=arm,other13_bitwise=True)

    def diagnostic(self):
        t=self.t;stacks=[]
        for name,m in self.m.named_modules():
            if not name or not (name.startswith('blocks.') or name.startswith('condition_embedder.text_embedder')):continue
            stack=[];stacks.append(stack)
            def pre(_m,_args,_name=name,_stack=stack):
                ctx=t.profiler.record_function('CHC.MODEL.'+_name);ctx.__enter__();_stack.append(ctx)
            def post(_m,_args,_out,_stack=stack):
                if _stack:_stack.pop().__exit__(None,None,None)
            self.hooks.extend([m.register_forward_pre_hook(pre),m.register_forward_hook(post,always_call=True)])
        prof=t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,t.profiler.ProfilerActivity.CUDA],
            record_shapes=False,with_stack=False,profile_memory=False)
        try:
            prof.start();rows=[self.sequence(a,timed=True,diagnostic=True,request='ON-'+a) for a in ['R','HC']]
            self.sync();prof.stop();trace=self.job/'traces/worker.json';prof.export_chrome_trace(str(trace))
            if trace.stat().st_size>2**30:raise ValueError('diagnostic trace above1GiB')
        finally:
            for h in self.hooks:h.remove()
            self.hooks=[]
            for stack in stacks:
                while stack:stack.pop().__exit__(None,None,None)
            self.control.diagnostic=False
        return dict(status='PASS',rows=rows,trace=str(trace),bytes=trace.stat().st_size,post_clean_OFF_only=True)

    def dispatch(self,command):
        key=command['stage'];self.ledger.begin(key);op=command['op']
        if op=='capture':out=self.capture()
        elif op=='resource_probe':out=self.probe()
        elif op=='sequence':out=self.sequence(command['arm'],timed=command['timed'],request=key)
        elif op=='invalidation':out=self.invalidation(command['case'])
        elif op=='isolation':out=self.isolation(command['arm'])
        elif op=='kernel':
            from prefix_micro import run_micro
            out=run_micro(self)
        elif op=='finish':
            rows=[self.sequence(a,subset=self.queries[-1:],request='finish-'+a) for a in ['R','S']]
            self.control.close();out=dict(status='PASS',rows=rows,all_caches_released=True)
        else:raise ValueError('unsupported bounded command')
        out.update(ledger=self.ledger.summary(),stage=key,strict_all_S_numerical_checks=self.strict_all_S_checks);return out

    def serve(self):
        index=0
        with self.termination_handler():
            try:
                write(self.base/'SERVING.json',dict(pid=os.getpid(),CPU_probe=self.cpu,termination_handler_installed=True),exclusive=True)
                while True:
                    self.check_stop();path=self.base/'inbox'/f'{index:04d}.json'
                    if not path.exists():time.sleep(.02);continue
                    command=json.loads(path.read_text())
                    if command['id']!=index or command['job']!=str(self.job):raise ValueError('command/job identity')
                    try:out=self.dispatch(command)
                    except Exception as exc:
                        out=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),ledger=self.ledger.summary())
                        write(self.base/'outbox'/path.name,out,exclusive=True);raise
                    if self.cpu and self.p.get('CPU_test_fault')=='pre_ack':self.cpu_pause('pre_ack')
                    write(self.base/'outbox'/path.name,out,exclusive=True)
                    write(self.base/'budget.json',self.ledger.summary())
                    if command['op']=='finish' or self.cpu and self.p.get('CPU_exit_after_one_command'):return
                    index+=1
            finally:
                if self.control is not None:self.control.close()
                self.close_counter('signal:'+signal.Signals(self._stop_signal).name if self._stop_signal else 'serve_exit')
                write(self.base/'EXIT.json',dict(timestamp=stamp(),ledger=self.ledger.summary(),CPU_probe=self.cpu,
                                                stop_signal=self._stop_signal))


def setup_cpu_runtime(job,p):
    # Same worker entry, Runtime, ledger, adapter and sequence method; actual
    # public tiny model. No production constructor/target weights/CUDA.
    import tempfile
    import test_cpu as fixture
    from capture import Query
    from production_fixture import loaded_model
    t=fixture.torch;dtype=t.bfloat16 if p.get('CPU_fixture_dtype')=='bfloat16' else t.float32
    m=loaded_model(dtype);qs=[]
    with t.no_grad():
        for i in range(29):
            inp=fixture.inputs(i,dtype=dtype);out=fixture.candidate(m,inp,t.tensor(fixture.ROWS));qs.append(Query(inp,dict(action_mode=True,update_cache=0,cache_name='pos',train_mode=False),out))
    s=fixture.NS(transformer=m,device=t.device('cpu'),cache_name='pos')
    rt=Runtime(s,None,job,p,CPU_probe=True);rt.queries=qs;rt.history=fixture.cache_state(m)
    rt.install_control();return rt


def cpu_probe(job,p):
    import test_cpu as fixture
    t=fixture.torch;rt=setup_cpu_runtime(job,p);results=[]
    try:
        with t.no_grad():
            for arm in ['R','S']:
                results.append(rt.dispatch(dict(op='sequence',stage='warmup_'+arm,arm=arm,timed=True)))
    finally:rt.control.close();rt.close_counter('CPU_probe_complete')
    if t.cuda.is_initialized() or fixture.CUDA_ATTEMPTS:raise ValueError('CPU worker initialized CUDA')
    write(Path(job)/'preflight/worker-CPU-probe.json',dict(status='PASS',same_actual_worker_entry_Runtime=True,
        real_all30_CPU=True,production_loader='modules.utils.load_transformer',
        production_model_module=type(rt.m).__module__,source_admission_before_capture=True,
        results=results,CUDA_initialized=False,target_weights_loaded=False,
        recovered_counter=recover(rt.base/'counter.jsonl',rt.counter_identity)),exclusive=True)


def cpu_serve(job,p):
    import test_cpu as fixture
    t=fixture.torch;rt=setup_cpu_runtime(job,p);hook=None
    if p.get('CPU_test_fault')=='mid_forward_2':
        def pause(_m,_args,_out):
            if rt.ledger.attempted==2:rt.cpu_pause('mid_forward_2')
        hook=rt.m.blocks[15].register_forward_hook(pause)
    try:
        with t.no_grad():rt.serve()
    except WorkerStop:pass
    finally:
        if hook is not None:hook.remove()
        if t.cuda.is_initialized() or fixture.CUDA_ATTEMPTS:raise ValueError('CPU worker initialized CUDA')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--cpu-probe',action='store_true');modes.add_argument('--cpu-serve',action='store_true')
    a=ap.parse_args()
    if a.cpu_probe or a.cpu_serve:
        p=read_cpu_plan(a.job)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU visibility')
        (cpu_probe if a.cpu_probe else cpu_serve)(Path(a.job),p);return
    # Claim before public server/Torch/CUDA imports; no CPU fault fields allowed.
    p=gpu_entry_gate(a.job,claim=True)
    sys.path.insert(0,p['worktree'])
    from wan_va import wan_va_server as source
    source.run_async_server_mode=lambda server,*_:Runtime(server,source,a.job,p).serve()
    source.init_logger();source.run(argparse.Namespace(config_name='libero',port=p['ports'][0],save_root=str(Path(a.job)/'worker/model_output')))


if __name__=='__main__':main()
