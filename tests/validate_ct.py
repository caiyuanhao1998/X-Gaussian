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
    args=parser.parse_args()
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
    temporary=tempfile.TemporaryDirectory()
    data=CTData(args.source_path, cache_dir=temporary.name) if args.source_path else phantom_data()
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
    zero_volume = query(pc,[0,0,0],[17]*3,[1.2]*3)['vol']
    parameters=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
    zero_gradients=torch.autograd.grad((zero_volume*0).sum(),parameters)
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g)==0 for g in zero_gradients)
    print('PASS zero-gradient voxel blocks return zero for all original attributes',flush=True)
    if args.source_path:
        from train import initialize
        initialized=initialize(data,3)
        expected=torch.from_numpy(data.initialization()[:,3:4]).float().cuda()
        assert torch.allclose(attenuation(initialized),expected,rtol=2e-5,atol=2e-7)
        print('PASS FDK density represented by existing X opacity/radiative attributes',flush=True)



def check_fdk():
    from ct_phantom import phantom_data
    from xct.initialization import fdk_volume
    data = phantom_data()
    cam = data.train[0]
    rows, cols = torch.meshgrid(torch.arange(512,device='cuda'), torch.arange(512,device='cuda'), indexing='ij')
    local = torch.stack([(cols+.5-256)*4/512, (rows+.5-256)*4/512, torch.full_like(cols,7)],-1).float()
    rotation = torch.linalg.inv(cam.world_view_transform.T)[:3,:3]
    direction = local @ rotation.T
    direction = direction / direction.norm(dim=-1,keepdim=True)
    origin = cam.camera_center
    closest_squared = origin.square().sum() - (direction*origin).sum(-1).square()
    sigma=.18
    projection = math.sqrt(2*math.pi)*sigma*torch.exp(-.5*closest_squared.clamp_min(0)/sigma**2)
    for camera in data.train:camera.original_image=projection[None]
    data.scanner['nVoxel']=np.array([64,48,40])
    data.scanner['sVoxel']=np.array([2.,1.5,1.])
    volume=fdk_volume(data.train,data.scanner)
    axes=[(torch.arange(n,device='cuda')+.5)*size/n-size/2 for n,size in zip(volume.shape,data.scanner['sVoxel'])]
    grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
    interior=grid.square().sum(-1)<.25**2
    outside=grid.square().sum(-1)>.6**2
    reference=torch.exp(-.5*grid.square().sum(-1)/sigma**2)
    assert abs(float(volume[interior].mean()/reference[interior].mean())-1)<.04
    assert float((volume-reference).square().mean().sqrt())<.01
    assert float(volume[outside].abs().mean())<.02
    print('PASS training-only FDK on an analytic Gaussian and non-cubic grid',flush=True)
    import pickle
    raw=dict(DSO=5000.,DSD=7000.,nVoxel=[64,48,40],dVoxel=[31.25,31.25,25.],
             nDetector=[512,512],dDetector=[7.8125,7.8125],offOrigin=[0,0,0],
             offDetector=[0,0],mode='cone',image=np.full((64,48,40),np.nan,dtype=np.float32),
             train=dict(angles=np.array([0.]),projections=projection.cpu().numpy()[None]))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'original.pickle'
        with path.open('wb') as stream:pickle.dump(raw,stream)
        original=CTData(path,load_test=False)
        assert float(original.scanner['DSO'])==5.
        assert np.allclose(original.scanner['sVoxel'],[2.,1.5,1.])
        assert torch.equal(original.train[0].original_image,projection[None])
        assert 'image' not in original.meta
    print('PASS original pickle scanner units and unused reference CT',flush=True)



def check_initialization_cache():
    from unittest.mock import patch
    points = np.array([[0., 0., 0., .1]], dtype=np.float32)
    for folder_format in [False, True]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'scene' if folder_format else root/'scene.pickle'
            source.mkdir() if folder_format else source.touch()
            data = CTData.__new__(CTData)
            data.path, data.cache_dir = source, root/'output'
            data.train, data.scanner = [], {}
            with patch('xct.initialization.fdk_points', return_value=points) as fdk:
                assert np.array_equal(data.initialization(), points)
                fdk.assert_called_once()
            with patch('xct.initialization.fdk_points', side_effect=AssertionError('Repeated FDK')):
                assert np.array_equal(data.initialization(), points)
                supplied = points * 2
                path = source/'init_scene.npy' if folder_format else root/'init_scene.npy'
                np.save(path, supplied)
                assert np.array_equal(data.initialization(), supplied)
    print('PASS automatic FDK and initialization cache for pickle and folder datasets', flush=True)


def check_thin_gaussian():
    from utils.general_utils import build_rotation
    pc = single()
    with torch.no_grad():
        pc._scaling[0] = torch.tensor([.008, .08, .18], device='cuda').log()
        pc._rotation[0] = torch.tensor([1., .2, -.3, .4], device='cuda')
    volume = query(pc, [0, 0, 0], [97, 61, 79], [1.2]*3)['vol']
    axes = [(torch.arange(n, device='cuda', dtype=torch.float64)+.5)*1.2/n-.6 for n in volume.shape]
    xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
    rotation = build_rotation(pc.get_rotation).double()[0]
    covariance = rotation @ torch.diag(pc.get_scaling.double()[0]**2) @ rotation.T
    reference = attenuation(pc).double().item()*torch.exp(-.5*torch.einsum('...i,ij,...j->...', xyz, covariance.inverse(), xyz))
    interior = (xyz.abs().amax(-1) < .53) & (reference > 2e-6)
    assert torch.allclose(volume[interior].double(), reference[interior], rtol=.002, atol=2e-6)
    gradients = torch.autograd.grad(volume.square().mean(),
        [pc._xyz, pc._features_dc, pc._features_rest, pc._opacity, pc._scaling, pc._rotation])
    assert all(torch.isfinite(g).all() for g in gradients)
    print('PASS thin rotated Gaussian retains above-threshold support and finite attribute gradients', flush=True)
    with torch.no_grad():
        pc._opacity.fill_(-30)
    empty = query(pc, [0, 0, 0], [17]*3, [1.2]*3)['vol']
    assert torch.count_nonzero(empty) == 0
    gradients = torch.autograd.grad(empty.sum(),
        [pc._xyz, pc._features_dc, pc._features_rest, pc._opacity, pc._scaling, pc._rotation])
    assert all(torch.count_nonzero(g) == 0 for g in gradients)
    print('PASS all Gaussians below the voxel threshold give a zero field and zero gradients', flush=True)


if __name__=='__main__':
    main()
    check_fdk()
    check_initialization_cache()
    check_thin_gaussian()
