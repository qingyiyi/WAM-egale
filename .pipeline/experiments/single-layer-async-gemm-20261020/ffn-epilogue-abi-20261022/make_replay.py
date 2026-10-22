"""Generate a measurement-only FFN entry from the selected exact source."""
from pathlib import Path
import hashlib,json,sys
B=Path(sys.argv[2]);B.mkdir(exist_ok=True)
source=Path(sys.argv[1]);s=source.read_text()
def sub(a,b):
 global s
 assert s.count(a)==1,(a[:80],s.count(a))
 s=s.replace(a,b)
sub('    int64_t *ffn_trace;', '    bool ffn_only;\n    int64_t *ffn_trace;')
sub('    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。', '''    if (p.ffn_only) {
        g.sync(); // Same counter visibility before standalone FFN.
        action_ffn_pipeline(p,s,g);
        return;
    }
    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。''')
sub('        const std::vector<Tensor> &context_kv) {','        const std::vector<Tensor> &context_kv, const std::vector<Tensor> &ffn_inputs) {')
sub('    p.debug_modulation = debug;', '    p.debug_modulation = debug;\n    p.ffn_only = !ffn_inputs.empty();')
sub('    // slot表和任务计数属于执行所需状态；逐tile visits/stats审计已移出核心。', '''    if(p.ffn_only) {
        b[M3]=ffn_inputs[0]; p.a[M3]=b[M3].data_ptr();
        b[R2]=ffn_inputs[1]; p.a[R2]=b[R2].data_ptr();
    }
    // slot表和任务计数属于执行所需状态；逐tile visits/stats审计已移出核心。''')
sub('          py::arg("context_kv") = std::vector<at::Tensor>{});', '''          py::arg("context_kv") = std::vector<at::Tensor>{},
          py::arg("ffn_inputs") = std::vector<at::Tensor>{});''')
(B/'replay.cu').write_text(s)
R=Path('算子研究/wam_ops/optimized')
build=(R/'build_whole_layer.py').read_text().replace('root=Path(__file__).resolve().parent',"root=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')")
build=build.replace("name='wam_whole_fa2_cuda',sources=[str(root/'csrc/whole_layer.cu')]", "name='wam_ffn_replay_cuda',sources=[str(Path(__file__).parent/'replay.cu')]")
(B/'build_replay.py').write_text("import sys\nsys.path.insert(0,'/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')\n"+build)
(B/'REPLAY-SOURCE-MANIFEST.json').write_text(json.dumps(dict(selected_source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),diagnostic_source_SHA256=hashlib.sha256(s.encode()).hexdigest(),scope='same full kernel with FFN-only branch and real M3/R2 aliases; full normal-path overhead and resource comparison required; no timestamps added'),indent=2)+'\n')
