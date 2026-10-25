"""核对本次HW派生文档、历史原文与运行源的边界；不读取模型文件。"""
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / '.pipeline'
M = Path(__file__).resolve().parent
reg = json.loads((M / 'REGISTRATION.json').read_text())
manifest = json.loads((M / 'WRITE-MANIFEST.json').read_text())
active = [P / item for item in manifest['written']]
broken, links = [], 0
for path in active:
    for href in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if re.match(r'^(?:[a-zA-Z][\w+.-]*:|#)', href):
            continue
        target = href.split('#', 1)[0]
        if not target:
            continue
        links += 1
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            broken.append({'file': str(path.relative_to(ROOT)), 'href': href})

protected = []
for rel in reg['protected']:
    protected.append({'path': rel, 'equal': (P / rel).read_bytes() == (M / 'protected-before' / rel).read_bytes()})
sources = []
for rel in reg['running_sources']:
    sources.append({'path': rel, 'equal': (ROOT / rel).read_bytes() == (M / 'source-before' / rel).read_bytes()})
archives = []
for rel in manifest['cycle_archives']:
    name = Path(rel).name.replace('.before-HW-compact-20261025', '')
    before = M / 'before' / Path(rel).parent / name
    assert (P / rel).read_bytes() == before.read_bytes(), rel
    archives.append(rel)
assert (P / manifest['old_index']).read_bytes() == (M / 'before/memory/INDEX.md').read_bytes()
old_records = list((M / 'before/memory/global').rglob('*.md'))
policy = 'requirements/decision-HW版本与读取策略.md'
for old in old_records:
    rel = old.relative_to(M / 'before/memory/global')
    now = P / 'memory/global' / rel
    assert now.exists(), str(rel)
    if str(rel) != policy:
        assert old.read_bytes() == now.read_bytes(), str(rel)

cycle = P / 'cycles/wam-research-20261016'
plan = (cycle / 'PLAN.md').read_text()
progress = (cycle / 'PROGRESS.md').read_text()
for field in ('current', 'next'):
    value = re.search(r'^' + field + r':\s*[\"\']?([^\"\'\s]+)', progress, re.M).group(1)
    assert value in plan, (field, value)
for path in (P / 'PROGRESS.compact.md', cycle / 'PROGRESS.md'):
    text = path.read_text()
    for flag in ('engineering_quality_screen', 'independent_test_complete', 'noninferiority_established', 'formal_performance_permission'):
        assert re.search(re.escape(flag) + r'\s*[:=]\s*false', text), (path, flag)

rels = [item['path'] for item in reg['old_default_files']] + ['memory/CURRENT-CONSTRAINTS.md']
sizes = [{'path': rel, 'bytes': (P / rel).stat().st_size} for rel in rels]
new_bytes = sum(item['bytes'] for item in sizes)
cycle_bytes = sum(item['bytes'] for item in sizes if item['path'].startswith('cycles/'))
assert new_bytes <= 24 * 1024
assert cycle_bytes <= 16 * 1024
assert (P / 'PROGRESS.compact.md').stat().st_size <= 8 * 1024
for topic in (P / 'memory/topics').glob('*.md'):
    assert topic.stat().st_size <= 4 * 1024, str(topic)
assert not broken, broken
assert all(item['equal'] for item in protected + sources)
result = {
    'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'status': 'PASS',
    'old_default_bytes': reg['old_default_bytes'],
    'new_default_bytes_including_new_constraint_view': new_bytes,
    'reduction_percent': 100 * (1 - new_bytes / reg['old_default_bytes']),
    'cycle_bytes': cycle_bytes,
    'memory_index_bytes': (P / 'memory/INDEX.md').stat().st_size,
    'files': sizes,
    'links_checked': links,
    'broken_links': broken,
    'protected': protected,
    'running_sources': sources,
    'byte_exact_history': archives,
    'old_memory_records': len(old_records),
    'remaining_memory_records': len(list((P / 'memory/global').rglob('*.md'))),
    'unchanged_memory_records': len(old_records) - 1,
    'updated_existing_memory': policy,
    'code_or_experiment_changes': False,
    'deleted_files': 0,
}
(M / 'VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ('status', 'old_default_bytes', 'new_default_bytes_including_new_constraint_view', 'reduction_percent', 'cycle_bytes', 'links_checked', 'remaining_memory_records')}, ensure_ascii=False))
