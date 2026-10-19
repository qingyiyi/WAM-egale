"""Host routing for complete Object episodes; fixed model math is unchanged."""
from pathlib import Path
import json
import runpy
import sys

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
role = sys.argv[1]
if role not in ('server', 'client'):
    raise ValueError('only actual server/client')
if role == 'server':
    from baai_resume_v1.path_binding import install
    install()
    from tf_suite_pipeline_v1 import collector as collection_module
    NativeCollector = collection_module.Collector
    class IsolatedAttemptCollector(NativeCollector):
        def __init__(self, server, proxy, output_dir, metadata):
            base = Path(output_dir).resolve()
            job = Path(metadata["job"]).resolve()
            if not job.is_relative_to((Path.home() / "cxz").resolve()):
                raise ValueError("BAAI new attempt must stay under ~/cxz")
            isolated = base / ("new-capture-" + job.name)
            isolated.mkdir(parents=True, exist_ok=False)
            super().__init__(server, proxy, isolated, metadata)
            if self.output != isolated or not self.output.is_relative_to(base):
                raise ValueError("Only filename namespace may change")
    collection_module.Collector = IsolatedAttemptCollector
    from tf_suite_pipeline_v1 import server_entry
    original = server_entry.run_rows
    def remaining_rows(plan):
        saved = set(plan.get('migration_readonly_completed_row_ids', []))
        return [row for row in original(plan) if row['row_id'] not in saved]
    server_entry.run_rows = remaining_rows
    sys.argv = [str(OWN / 'entry.py'), *sys.argv[2:]]
    raise SystemExit(server_entry.main())
script = OWN.parent / 'tf_suite_pipeline_v1/client_entry.py'
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name='__main__')
