"""Independent analytic phantom check for world-to-voxel ray integration."""
import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,numpy as np
from validate_ct import single
from ct_phantom import phantom_data
from xct.field import query,attenuation
from xct.rays import project_volume
pc=single();scales=torch.tensor([.12,.2,.28],device='cuda')
with torch.no_grad():pc._scaling[0]=scales.log()
data=phantom_data()
vol=query(pc,[0,0,0],[128]*3,[2.]*3)['vol'];indices=torch.tensor([256*512+256],device='cuda')
for index in [0,12,25,37]:
    cam=data.train[index]
    c2w=torch.linalg.inv(cam.world_view_transform.T)
    local=torch.tensor([.5*4/512,.5*4/512,7.],device='cuda')
    d=(c2w[:3,:3]@local).double();d=d/d.norm();o=cam.camera_center.double();s=scales.double()
    aa=((d/s)**2).sum();bb=(o*d/s**2).sum();cc=((o/s)**2).sum()
    expected=attenuation(pc).item()*math.sqrt(2*math.pi/aa.item())*math.exp(-.5*(cc-bb*bb/aa).item())
    value=project_volume(vol,cam,data.scanner,indices,512)
    assert abs(value.item()/expected-1)<.015,(index,value.item(),expected)
    print('PASS anisotropic Gaussian ray / camera',index,value.item(),expected)
grad=torch.autograd.grad(value.sum(),[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation])
assert all(torch.isfinite(g).all() for g in grad)
print('PASS end-to-end voxel/ray gradients to six original attribute groups')
cam=data.train[0];idx=torch.tensor([256*512+256],device='cuda')
c2w=torch.linalg.inv(cam.world_view_transform.T)
local=torch.tensor([.5*4/512,.5*4/512,7.],device='cuda');direction=c2w[:3,:3]@local;direction=direction/direction.norm();origin=cam.camera_center
closest=origin-direction*torch.dot(origin,direction)
n=128;cell=torch.floor((closest+1)*n/2).long().clamp(0,n-1)
volume=torch.zeros(n,n,n,device='cuda',requires_grad=True)
with torch.no_grad():volume[tuple(cell)]=1
pred=project_volume(volume,cam,data.scanner,idx)
low=-1+cell.double()*2/n;high=low+2/n
t1=(low-origin.double())/direction.double();t2=(high-origin.double())/direction.double()
expected=(torch.maximum(t1,t2).min()-torch.minimum(t1,t2).max()).clamp_min(0)
assert abs(pred.item()-expected.item())<2e-6,(pred.item(),expected.item())
grad=torch.autograd.grad(pred.sum(),volume)[0]
assert abs(grad[tuple(cell)].item()-expected.item())<2e-6
print('PASS exact one-voxel intersection length and linear adjoint',pred.item(),expected.item())
