"""Training-only FDK initialization at original voxel resolution; never read CT truth."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import tigre
import tigre.algorithms as algs
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--views',choices=['trainval','full'],default='trainval');p.add_argument('--n_points',type=int,default=50000);a=p.parse_args()
if a.n_points<1:raise ValueError('n_points must be positive')
source=Path(a.source);out=Path(a.output)
meta=json.loads((source/'meta_data.json').read_text());cfg=meta['scanner'];scale=2/max(cfg['sVoxel'])
frames=[f for i,f in enumerate(meta['proj_train']) if a.views=='full' or i%5!=0]
projs=np.stack([np.load(source/f['file_path']) for f in frames]).astype(np.float32)*scale
angles=np.array([f['angle'] for f in frames])
geo=tigre.geometry(mode='cone');geo.DSO=cfg['DSO']*scale;geo.DSD=cfg['DSD']*scale
geo.nDetector=np.array(cfg['nDetector']);geo.sDetector=np.array(cfg['sDetector'])*scale;geo.dDetector=geo.sDetector/geo.nDetector
geo.nVoxel=np.array(cfg['nVoxel'][::-1]);geo.sVoxel=np.array(cfg['sVoxel'][::-1])*scale;geo.dVoxel=geo.sVoxel/geo.nVoxel
geo.offOrigin=np.array(cfg['offOrigin'][::-1])*scale;geo.offDetector=np.array([cfg['offDetector'][1],cfg['offDetector'][0],0])*scale
geo.accuracy=cfg['accuracy'];geo.filter=None
volume=algs.fdk(projs[:,::-1,:],geo,angles).transpose(2,1,0)
valid=np.argwhere(volume>.05)
if len(valid)<a.n_points:raise ValueError(f'Only {len(valid)} eligible initialization voxels')
np.random.seed(0);idx=valid[np.random.choice(len(valid),a.n_points,replace=False)]
xyz=idx*(np.array(cfg['sVoxel'])*scale/np.array(cfg['nVoxel']))-np.array(cfg['sVoxel'])*scale/2+np.array(cfg['offOrigin'])*scale
points=np.concatenate([xyz,volume[idx[:,0],idx[:,1],idx[:,2],None]*.15],1)
assert np.isfinite(points).all();out.parent.mkdir(parents=True,exist_ok=True);np.save(out,points)
out.with_suffix('.json').write_text(json.dumps(dict(training_views=[f['file_path'] for f in frames],internal_split=a.views,validation_views_used=a.views=='full',official_test_used=False,ground_truth_used=False,seed=0,points=len(points),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
print('Saved training-only FDK initialization',points.shape,flush=True)
