"""Source-preserving adapter for the pinned Base infer_action implementation."""
import ast,copy,inspect,textwrap,types
from pathlib import Path
import torch

def source_function(path,name,namespace):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]

def install(model,source,output):
    original=type(model).infer_action
    namespace=dict(inspect.unwrap(original).__globals__)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    func=tree.body[0]
    positions=[i for i,n in enumerate(func.body) if isinstance(n,ast.For)]
    assert len(positions)==1,positions
    index=positions[0]
    assert isinstance(func.body[index].target,ast.Tuple)
    assert [x.id for x in func.body[index].target.elts]==['step_t_action','step_delta_action']
    replacement=ast.parse('''latents_action = self.empirical_replay.loop(latents_action, infer_timesteps_action, infer_deltas_action, context, context_mask, video_cache_k, video_cache_v, action_attention_mask, denoise_action_with_video_cache, compile_action_infer)''').body
    replay=copy.deepcopy(func);replay.name='infer_action_replay';replay.body[index:index+1]=replacement
    prepare=copy.deepcopy(func);prepare.name='prepare_action_inputs'
    prepare.body=prepare.body[:index]+ast.parse('''return dict(state=latents_action, timesteps=infer_timesteps_action, deltas=infer_deltas_action, ctx=context, mask=context_mask, cache_k=video_cache_k, cache_v=video_cache_v, attn=action_attention_mask, fn=denoise_action_with_video_cache, compiled=compile_action_infer)''').body
    # Phase timers are only used in separate diagnostic calls, never the full-chunk comparison.
    profiled=[]
    for base in (copy.deepcopy(func),copy.deepcopy(replay)):
        base.name+='__profiled'
        def phase(node):
            s=ast.unparse(node)
            if '_encode_input_image_latents_tensor' in s:return 'vae'
            if 'if use_prompt:' in s:return 'text'
            if 'self._append_proprio_to_context' in s:return 'proprio'
            if 'self.video_expert.prepare(' in s:return 'visual_prepare'
            if s.startswith('video_cache_k, video_cache_v = prefill_video_cache('):return 'visual_prefill'
            if 'self.infer_action_scheduler.build_inference_schedule(' in s:return 'scheduler_grid'
            if isinstance(node,ast.For) or 'self.empirical_replay.loop(' in s:return 'action_denoise'
            if 'latents_action = torch.randn(' in s:return 'noise_initialization'
        body=[]
        for node in base.body:
            p=phase(node)
            if p:body.append(ast.With(items=[ast.withitem(context_expr=ast.parse(f'self.phase_clock({p!r})',mode='eval').body)],body=[node],type_comment=None))
            else:body.append(node)
        base.body=body;profiled.append(base)
    module=ast.fix_missing_locations(ast.Module(body=[replay,prepare]+profiled,type_ignores=[]))
    output=Path(output);output.write_text(ast.unparse(module)+'\n')
    exec(compile(module,str(output),'exec'),namespace)
    for n in module.body:setattr(model,n.name,types.MethodType(namespace[n.name],model))
    return original

@torch.no_grad()
def canonical_action(model,state,timesteps,deltas,ctx,mask,cache_k,cache_v,attn,fn,compiled):
    # Identical canonical loop; final scheduler output is not graph-owned.
    for t,d in zip(timesteps,deltas):
        if compiled:torch.compiler.cudagraph_mark_step_begin()
        v=fn(latents_action=state,timestep_action=t.unsqueeze(0).to(dtype=state.dtype,device=model.device),context=ctx,context_mask=mask,video_cache_k=cache_k,video_cache_v=cache_v,action_attention_mask=attn)
        state=model.infer_action_scheduler.step(v,d,state)
    return state
