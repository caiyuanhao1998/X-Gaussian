"""CUDA Siddon integration, differentiable with respect to the voxel field."""
import torch
from x_gaussian_ct_rasterization import _rays

class Integrate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,volume,origin,directions,low,spacing):
        ctx.save_for_backward(volume,origin,directions,low,spacing)
        return _rays.forward(volume,origin,directions,low,spacing)
    @staticmethod
    def backward(ctx,gradient):
        volume,origin,directions,low,spacing=ctx.saved_tensors
        result=_rays.backward(gradient.contiguous(),volume,origin,directions,low,spacing)
        return result,None,None,None,None

def project_volume(volume,camera,scanner,indices):
    dtype,device=volume.dtype,volume.device
    rows=torch.div(indices,camera.image_width,rounding_mode='floor').to(dtype)
    cols=(indices%camera.image_width).to(dtype)
    detector=torch.as_tensor(scanner['sDetector'],device=device,dtype=dtype)
    local=torch.stack([(cols+.5-camera.image_width/2)*detector[1]/camera.image_width,
                       (rows+.5-camera.image_height/2)*detector[0]/camera.image_height,
                       torch.full_like(rows,float(scanner['DSD']))],1)
    c2w=torch.linalg.inv(camera.world_view_transform.T).to(dtype)
    direction=local@c2w[:3,:3].T
    direction=direction/torch.linalg.vector_norm(direction,dim=1,keepdim=True)
    size=torch.as_tensor(scanner['sVoxel'],device=device,dtype=dtype)
    center=torch.as_tensor(scanner['offOrigin'],device=device,dtype=dtype)
    spacing=size/torch.as_tensor(volume.shape,device=device,dtype=dtype)
    return Integrate.apply(volume.contiguous(),camera.camera_center.to(dtype).contiguous(),
                           direction.contiguous(),(center-size/2).contiguous(),spacing.contiguous())
