"""Run the unchanged server/client behind the explicit relocation contract."""
from pathlib import Path
import runpy,sys
OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from baai_approved_retry_v2.contract import install
install()
role=sys.argv[1]
if role=='server':
    from baai_resume_v1.path_binding import install as bind_legacy_root
    bind_legacy_root()
if role not in ('server','client'):raise ValueError('registered server/client role only')
script=OWN.parent/'tf_fullsuite_eval_v2'/('server_entry.py' if role=='server' else 'client_entry.py')
sys.argv=[str(script),*sys.argv[2:]]
runpy.run_path(str(script),run_name='__main__')
