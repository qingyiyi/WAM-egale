"""Relocate only the four frozen URDF/collision file paths; geometry/numbers untouched."""
from pathlib import Path
import difflib,json,time
P=Path('/root/autodl-tmp/cxz/spec-speed-20261025');root=P/'sim/RoboTwin'
before='/root/cxz/robotwin/source/RoboTwin-pinned/'
rows=[];diffs=[]
for file in [root/'assets/embodiments/aloha-agilex/curobo_left.yml',root/'assets/embodiments/aloha-agilex/curobo_right.yml']:
    old=file.read_text();assert old.count(before)==2
    new=old.replace(before,str(root)+'/')
    for line in new.splitlines():
        if line.lstrip().startswith(('urdf_path:','collision_spheres:')):
            value=line.split(':',1)[1].strip();assert Path(value).is_file(),value
    diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(file.relative_to(P))+':original',tofile=str(file.relative_to(P))+':5090-paths'))
    file.write_text(new);rows.append(str(file.relative_to(P)))
(P/'results/ROBOTWIN-RESOURCE-PATHS.patch').write_text(''.join(diffs))
(P/'results/ROBOTWIN-RESOURCE-RELOCATION.json').write_text(json.dumps(dict(at=time.time(),files=rows,only_resource_prefix_replaced=True,geometry_and_numeric_text_unchanged=True),indent=2)+'\n')
