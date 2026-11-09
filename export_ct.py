"""Voxelize a CT-trained X PLY without projections, FDK, or ground-truth volume."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scene.gaussian_model import GaussianModel_Xray
from xct.field import query,configure_field
from xct.io import field_metadata

p=argparse.ArgumentParser()
p.add_argument('--model',required=True,help='CT-trained point_cloud.ply')
p.add_argument('--output',required=True,help='Output .npy volume')
p.add_argument('--resolution',type=int,nargs=3,default=[256,256,256])
p.add_argument('--size',type=float,nargs=3,default=[2,2,2])
p.add_argument('--center',type=float,nargs=3,default=[0,0,0])
p.add_argument('--sh_degree',type=int)
p.add_argument('--activation',choices=['relu','softplus','softplus_sum'])
p.add_argument('--beta',type=float)
p.add_argument('--backend',choices=['standard','precision'])
a=p.parse_args()
if min(a.resolution)<=0 or min(a.size)<=0:p.error('Resolution and size must be positive')
model=Path(a.model).resolve()
metadata=model.parents[2]/'config.json' if len(model.parents)>2 else None
config=json.loads(metadata.read_text()) if metadata is not None and metadata.exists() else {}
config.update(field_metadata(a.model))
activation=a.activation or config.get('activation','relu')
beta=a.beta if a.beta is not None else config.get('beta',1.)
degree=a.sh_degree if a.sh_degree is not None else config.get('sh_degree',3)
backend=a.backend or config.get('backend','standard')
pc=GaussianModel_Xray(degree);pc.load_ply(a.model);configure_field(pc,activation,beta,backend)
with torch.no_grad():volume=query(pc,a.center,a.resolution,a.size)['vol']
if not torch.isfinite(volume).all():raise FloatingPointError('Nonfinite volume')
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
np.save(output,volume.cpu().numpy())
output.with_suffix('.json').write_text(json.dumps(dict(
    field='X CT attenuation decoded from original point attributes',activation=activation,beta=beta,backend=backend,
    axis_order='xyz',sample_location='voxel centers',resolution=a.resolution,
    size=a.size,center=a.center,spacing=(np.array(a.size)/a.resolution).tolist(),
    model=str(Path(a.model).resolve()),ground_truth_used=False),indent=2)+'\n')
print('Saved CT volume',tuple(volume.shape),'to',output)
