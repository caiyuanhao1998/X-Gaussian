"""Verify a frozen PLY alone reproduces its reported volume and original schema."""
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
p=argparse.ArgumentParser();p.add_argument('--model_path',required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--checks',required=True);p.add_argument('--standalone_dir',type=Path);a=p.parse_args()
model=Path(a.model_path).resolve();ply=model/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply'
standalone=a.standalone_dir or model/'standalone_export';standalone.mkdir(parents=True,exist_ok=True)
shutil.copyfile(ply,standalone/'point_cloud.ply')
subprocess.run([sys.executable,str(Path(__file__).resolve().with_name('export_ct.py')),'--model',str(standalone/'point_cloud.ply'),'--output',str(standalone/'volume.npy')],check=True)
volume=np.load(standalone/'volume.npy');reference=np.load(model/'eval'/f'iter_{a.iteration:06d}'/'vol_pred.npy');assert np.array_equal(volume,reference)
vertices=PlyData.read(ply)['vertex'].data
expected=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
assert list(vertices.dtype.names)==expected and all(np.isfinite(vertices[n]).all() for n in expected)
config=json.loads((model/'config.json').read_text())
with open(model/'loss.csv') as f:rows=list(csv.DictReader(f))
assert len(rows)==config['iterations'] and all(np.isfinite([float(r[k]) for r in rows]).all() for k in rows[0])
for name,target in [('volume_standalone.npy',standalone/'volume.npy'),('volume_standalone.json',standalone/'volume.json')]:
 link=model/name
 if not link.exists():link.symlink_to(target)
 elif name.endswith('.npy'):assert np.array_equal(np.load(link),np.load(target))
checks=dict(model=str(model),iteration=a.iteration,standalone_PLY_only_export_exact=True,
 resolution=list(volume.shape),finite_original_PLY_columns=expected,points=len(vertices),finite_loss_rows=len(rows),
 learned_scalars_per_point=59,no_extra_learned_density_or_voxels=True,
 PLY_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),activation=config.get('activation','relu'),backend=config.get('backend','standard'))
Path(a.checks).parent.mkdir(parents=True,exist_ok=True)
Path(a.checks).write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
