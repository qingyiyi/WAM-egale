"""External before/after timing copies; never instrument the active source."""
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parent
R = Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
sources = {'before': B.parent / 'cross/before/whole_layer.cu',
           'after': R / 'csrc/whole_layer.cu'}
manifest = []
for label, source in sources.items():
    s = source.read_text()
    original = s
    s = s.replace('wam_whole_layer_v1', 'wam_profile_' + label)
    s = s.replace('int *queues;', 'int *queues; int64_t *profile_ticks;')
    marker = '__device__ void linear('
    pos = s.index(marker)
    s = s[:pos] + '''__device__ inline int64_t diagnostic_ns() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
''' + s[pos:]
    for name in ('linear', 'context_linear', 'action_ffn_linear'):
        signature = 'void ' + name + '('
        if signature not in s:
            continue
        a = s.index('{', s.index(signature))
        level = 1
        z = a + 1
        while level:
            level += (s[z] == '{') - (s[z] == '}')
            z += 1
        body = s[a + 1:z - 1]
        assert body.rstrip().endswith('g.sync();')
        body = '\n    if (g.thread_rank()==0) p.profile_ticks[2*st]=diagnostic_ns();' + body
        body += '\n    if (g.thread_rank()==0) p.profile_ticks[2*st+1]=diagnostic_ns();\n'
        s = s[:a + 1] + body + s[z - 1:]
    s = s.replace('auto queues = at::empty({MAX_STAGES}, io);',
                  'auto queues = at::empty({MAX_STAGES}, io);\n'
                  '    auto ticks = at::empty({MAX_STAGES,2},lo); p.profile_ticks=ticks.data_ptr<int64_t>();')
    s = s.replace('result["cache_info"] = info;',
                  'result["cache_info"] = info; result["profile_ticks"] = ticks;')
    (B / ('profile_' + label + '.cu')).write_text(s)
    manifest.append(dict(label=label, source=str(source), SHA256=hashlib.sha256(original.encode()).hexdigest(),
                         instrumented_SHA256=hashlib.sha256(s.encode()).hexdigest()))
(B / 'SOURCE-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
