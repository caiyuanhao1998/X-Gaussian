"""CUDA/Torch forward and adjoint parity for axis rays, misses and inside origins."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xct.data import CTData,angle2pose
from xct.rays import project_volume as reference
from xct.cuda_rays import project_volume as fast
torch.manual_seed(716)
results=[]
def test(name,volume,camera,scanner,indices,timing=False):
    a=volume.detach().requires_grad_()
    b=volume.detach().requires_grad_()
    target=torch.randn(len(indices),device='cuda',dtype=volume.dtype)
    x=reference(a,camera,scanner,indices)
    y=fast(b,camera,scanner,indices)
    (x*target).sum().backward();(y*target).sum().backward()
    forward=float((x-y).abs().max())
    gradient=float((a.grad-b.grad).norm()/a.grad.norm().clamp_min(1e-12))
    limit=2e-10 if volume.dtype==torch.float64 else 3e-4
    assert forward<limit and gradient<limit,(name,forward,gradient)
    record=dict(name=name,shape=list(volume.shape),dtype=str(volume.dtype),rays=len(indices),
                forward_max_abs=forward,gradient_relative_l2=gradient)
    del a,b,x,y
    if timing:
        def measure(function):
            with torch.no_grad():
                function(volume,camera,scanner,indices)
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(5):function(volume,camera,scanner,indices)
                torch.cuda.synchronize()
                return (time.perf_counter()-start)/5
        old,new=measure(reference),measure(fast)
        record.update(torch_forward_seconds=old,cuda_forward_seconds=new,ray_forward_speedup=old/new)
    results.append(record)
    print('PASS',json.dumps(record),flush=True)

for dtype in [torch.float64,torch.float32]:
    pose=torch.tensor(angle2pose(2.,0.),device='cuda',dtype=dtype)
    camera=SimpleNamespace(image_height=5,image_width=7,world_view_transform=torch.linalg.inv(pose).T.contiguous(),camera_center=pose[:3,3].contiguous())
    scanner=dict(sDetector=[3.,4.],DSD=3.,sVoxel=[1.,1.,1.],offOrigin=[0.,0.,0.])
    indices=torch.arange(35,device='cuda')
    unit=torch.ones((17,19,23),device='cuda',dtype=dtype)
    assert abs(float(fast(unit,camera,scanner,indices)[17])-1.)<1e-6
    test('axis_aligned_and_misses',torch.rand_like(unit),camera,scanner,indices)
    pose=torch.eye(4,device='cuda',dtype=dtype)
    camera.world_view_transform=pose;camera.camera_center=pose[:3,3]
    assert abs(float(fast(unit,camera,scanner,indices)[17])-.5)<1e-6
    test('origin_inside',torch.rand_like(unit),camera,scanner,indices)

print('PASS CUDA/Torch forward and adjoint, axis rays, misses, origin inside, float32/64')
