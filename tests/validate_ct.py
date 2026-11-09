"""Numerical checks of the new physical field, geometry and gradient chain."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import math
import argparse
import tempfile
import numpy as np
import torch
from torch import nn
from scene.gaussian_model import GaussianModel_Xray
from utils.sh_utils import C0
from xct.data import CTData
from xct.field import attenuation,query,render


def single():
    pc=GaussianModel_Xray(3);pc.active_sh_degree=3
    pc._xyz=nn.Parameter(torch.zeros(1,3,device='cuda'))
    pc._features_dc=nn.Parameter(torch.full((1,1,3),(.8-.5)/C0,device='cuda'))
    pc._features_rest=nn.Parameter(torch.zeros(1,15,3,device='cuda'))
    pc._opacity=nn.Parameter(torch.tensor([[math.log(.2/.8)]],device='cuda'))
    pc._scaling=nn.Parameter(torch.full((1,3),math.log(.18),device='cuda'))
    pc._rotation=nn.Parameter(torch.tensor([[1.,0.,0.,0.]],device='cuda'))
    return pc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source_path',help='Optional original-resolution CT dataset for integration checks')
    parser.add_argument('--r2_repo',type=Path,help='Optional R2 checkout for exact camera comparison')
    args=parser.parse_args()
    if args.r2_repo and not args.source_path:parser.error('--r2_repo requires --source_path')
    torch.manual_seed(0)
    pc=single();rho=attenuation(pc).item()
    assert abs(rho-.16)<1e-6
    vol=query(pc,[0,0,0],[17]*3,[1.2]*3)['vol']
    grid=(torch.arange(17,device='cuda')+.5)*1.2/17-.6
    xyz=torch.stack(torch.meshgrid(grid,grid,grid,indexing='ij'),-1)
    ref=rho*torch.exp(-.5*(xyz**2).sum(-1)/.18**2)
    mask=(xyz.abs().amax(-1)<2.5*.18)  # Compare inside the CUDA three-sigma tile support.
    assert torch.allclose(vol[mask],ref[mask],rtol=2e-5,atol=2e-6),(vol-ref).abs().max()
    print('PASS analytic Gaussian voxel values',flush=True)
    from ct_phantom import phantom_data
    data=CTData(args.source_path) if args.source_path else phantom_data()
    cam=data.train[0]
    projection=render(cam,pc)['render']
    expected=rho*math.sqrt(2*math.pi)*.18
    assert abs(projection[0,256,256].item()/expected-1)<.002
    print('PASS analytic central-ray Gaussian integral',flush=True)
    # Scalar gradients flow through BOTH original X feature blocks and opacity.
    pc._xyz.data[0]=torch.tensor([.03,-.02,.01],device='cuda')
    pc._scaling.data[0]=torch.tensor([.14,.18,.22],device='cuda').log()
    pc._rotation.data[0]=torch.tensor([1.,.1,.2,-.05],device='cuda')
    voxel_weights=torch.rand(17,17,17,device='cuda')/100
    image_weights=torch.zeros_like(projection);image_weights[:,240:270,245:275]=torch.rand(1,30,30,device='cuda')/100
    for name,objective in [
        ('voxel',lambda:(query(pc,[0,0,0],[17]*3,[1.2]*3)['vol']*voxel_weights).sum()),
        ('projection',lambda:(render(cam,pc)['render']*image_weights).sum())]:
        parameters=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
        gradients=torch.autograd.grad(objective(),parameters)
        indices=[(0,0),(0,0,0),(0,3,1),(0,0),(0,1),(0,1)]
        for parameter,gradient,index in zip(parameters,gradients,indices):
            assert torch.isfinite(gradient).all()
            old=parameter[index].item();eps=.002
            with torch.no_grad():
                parameter[index]=old+eps;plus=objective().item()
                parameter[index]=old-eps;minus=objective().item()
                parameter[index]=old
            numeric=(plus-minus)/(2*eps);analytic=gradient[index].item()
            assert abs(numeric-analytic)<max(.0005,.025*abs(numeric)),(name,index,numeric,analytic)
        print('PASS '+name+' finite differences for all six original attribute groups',flush=True)
    # Direct PLY round trip preserves the exact original field schema and attenuation.
    with tempfile.TemporaryDirectory() as tmp:
        path=str(Path(tmp)/'point_cloud.ply');pc.save_ply(path)
        loaded=GaussianModel_Xray(3);loaded.load_ply(path)
        assert loaded.construct_list_of_attributes()==pc.construct_list_of_attributes()
        assert torch.allclose(attenuation(loaded),attenuation(pc))
    print('PASS original X PLY schema and round trip',flush=True)
    if args.r2_repo:
        # Optional external integration check; the default tests need no R2 install.
        sys.path.insert(0,str(args.r2_repo.resolve()))
        from r2_gaussian.dataset import Scene
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            other=Scene(SimpleNamespace(source_path=str(data.path),model_path=tmp,eval=True,data_device='cuda'),shuffle=False)
            for a,b in zip(data.train+data.test,other.getTrainCameras()+other.getTestCameras()):
                assert torch.equal(a.original_image,b.original_image)
                assert torch.allclose(a.world_view_transform,b.world_view_transform,atol=1e-6)
                assert torch.allclose(a.full_proj_transform,b.full_proj_transform,atol=1e-6)
        print('PASS exact original-resolution projections and R2-equivalent camera geometry',flush=True)
    if args.source_path:
        from train_ct import initialize
        initialized=initialize(data,3)
        expected=torch.from_numpy(data.initialization()[:,3:4]).float().cuda()
        assert torch.allclose(attenuation(initialized),expected,rtol=2e-5,atol=2e-7)
        print('PASS FDK density represented by existing X opacity/radiative attributes',flush=True)



if __name__=='__main__':main()
