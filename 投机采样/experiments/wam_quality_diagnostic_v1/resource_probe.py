"""One-shot NVML metadata child; the independent outer supervisor bounds it."""
import argparse,json,sys
from pathlib import Path
OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN.parent/'single_worker_chc_gpu_v4'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    from resource_guard import NVML
    nv=NVML()
    try:row=nv.sample()
    finally:nv.close()
    with a.out.open('x') as f:json.dump(row,f)

if __name__=='__main__':main()
