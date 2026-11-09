"""Differentiable cone-ray integration of the voxelized field (no CT supervision)."""
import torch
import torch.nn.functional as F


def project_volume_trilinear(volume,camera,scanner,indices,samples=384):
    device=volume.device;dtype=volume.dtype
    width=camera.image_width;height=camera.image_height
    rows=torch.div(indices,width,rounding_mode='floor').to(dtype)
    cols=(indices%width).to(dtype)
    detector=torch.tensor(scanner['sDetector'],device=device,dtype=dtype)
    dsd=float(scanner['DSD'])
    local=torch.stack([(cols+.5-width/2)*detector[1]/width,
                       (rows+.5-height/2)*detector[0]/height,
                       torch.full_like(rows,dsd)],1)
    c2w=torch.linalg.inv(camera.world_view_transform.T)
    direction=local@c2w[:3,:3].T
    direction=direction/torch.linalg.vector_norm(direction,dim=1,keepdim=True)
    origin=camera.camera_center
    center=torch.tensor(scanner['offOrigin'],device=device,dtype=dtype)
    size=torch.tensor(scanner['sVoxel'],device=device,dtype=dtype)
    inv=torch.where(direction.abs()>1e-8,1/direction,torch.full_like(direction,1e8))
    a=(center-size/2-origin)*inv;b=(center+size/2-origin)*inv
    near=torch.minimum(a,b).amax(1).clamp_min(0)
    far=torch.maximum(a,b).amin(1)
    length=(far-near).clamp_min(0)
    t=near[:,None]+length[:,None]*(torch.arange(samples,device=device,dtype=dtype)+.5)/samples
    xyz=origin[None,None,:]+direction[:,None,:]*t[:,:,None]
    normalized=(xyz-center)*2/size
    # volume axes are xyz; grid_sample coordinate order is W,H,D = z,y,x.
    grid=normalized.flip(-1)[None,:,:,None,:]
    values=F.grid_sample(volume[None,None],grid,mode='bilinear',padding_mode='zeros',align_corners=False).reshape(len(indices),samples)
    return values.mean(1)*length


def project_volume(volume,camera,scanner,indices,samples=384,method='siddon'):
    if method=='trilinear':return project_volume_trilinear(volume,camera,scanner,indices,samples)
    if method!='siddon':raise ValueError(method)
    # Exact path-length integration of this piecewise-constant field (not bitwise TIGRE).
    dtype=volume.dtype;device=volume.device
    rows=torch.div(indices,camera.image_width,rounding_mode='floor').to(dtype)
    cols=(indices%camera.image_width).to(dtype)
    size=torch.tensor(scanner['sVoxel'],device=device,dtype=dtype)
    center=torch.tensor(scanner['offOrigin'],device=device,dtype=dtype)
    detector=torch.tensor(scanner['sDetector'],device=device,dtype=dtype)
    local=torch.stack([(cols+.5-camera.image_width/2)*detector[1]/camera.image_width,
                       (rows+.5-camera.image_height/2)*detector[0]/camera.image_height,
                       torch.full_like(rows,float(scanner['DSD']))],1)
    c2w=torch.linalg.inv(camera.world_view_transform.T)
    direction=local@c2w[:3,:3].T
    direction=direction/torch.linalg.vector_norm(direction,dim=1,keepdim=True)
    origin=camera.camera_center
    inv=torch.where(direction.abs()>1e-8,1/direction,torch.full_like(direction,1e8))
    low=center-size/2;high=center+size/2
    a=(low-origin)*inv;b=(high-origin)*inv
    near=torch.minimum(a,b).amax(1).clamp_min(0)
    far=torch.maximum(a,b).amin(1)
    far=torch.maximum(far,near)
    crossings=[]
    for axis,n in enumerate(volume.shape):
        planes=low[axis]+torch.arange(n+1,device=device,dtype=dtype)*(size[axis]/n)
        t=(planes[None,:]-origin[axis])*inv[:,axis,None]
        crossings.append(torch.maximum(torch.minimum(t,far[:,None]),near[:,None]))
    t=torch.cat(crossings+[near[:,None],far[:,None]],1).sort(1).values
    length=t[:,1:]-t[:,:-1]
    mid=(t[:,1:]+t[:,:-1])*.5
    xyz=origin[None,None,:]+direction[:,None,:]*mid[:,:,None]
    dims=torch.tensor(volume.shape,device=device,dtype=dtype)
    ijk=torch.floor((xyz-low)*dims/size).long()
    ix=ijk[:,:,0].clamp(0,volume.shape[0]-1)
    iy=ijk[:,:,1].clamp(0,volume.shape[1]-1)
    iz=ijk[:,:,2].clamp(0,volume.shape[2]-1)
    flat=(ix*volume.shape[1]+iy)*volume.shape[2]+iz
    return (volume.reshape(-1)[flat]*length).sum(1)
