"""A view-independent attenuation field using the existing X-Gaussian attributes.

Matches the constant radiative-feature weights in the ORIGINAL X CUDA
computeColorFromSH (not the directional RGB SH function in utils/sh_utils).
No additional per-Gaussian density, latent, network or voxel parameters.
"""
import math
from functools import lru_cache
import torch
from utils.sh_utils import C0, C1, C2, C3
from x_gaussian_ct_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer,
    GaussianVoxelizationSettings, GaussianVoxelizer,
)


def configure_field(pc, activation='relu', beta=1., backend='standard'):
    if activation not in ['relu','softplus','softplus_sum']:raise ValueError(activation)
    if backend not in ['standard','precision']:raise ValueError(backend)
    pc.ct_backend=backend
    pc.ct_activation=activation
    pc.ct_beta=beta
    return pc


@lru_cache(maxsize=2)
def operators(backend):
    if backend=='standard':
        return GaussianRasterizationSettings,GaussianRasterizer,GaussianVoxelizationSettings,GaussianVoxelizer
    if backend=='precision':
        from x_gaussian_ct_precision_rasterization import (
            GaussianRasterizationSettings as RS,GaussianRasterizer as R,
            GaussianVoxelizationSettings as VS,GaussianVoxelizer as V)
        return RS,R,VS,V
    raise ValueError(backend)


def radiative_channels(pc):
    if not 0 <= pc.active_sh_degree <= 3:
        raise ValueError('X radiative feature degree must be between 0 and 3')
    count = (pc.active_sh_degree + 1) ** 2
    weights = pc.get_features.new_tensor([C0, -C1, C1, -C1] + C2 + C3)
    channels = 0.5 + (pc.get_features[:, :count] * weights[:count, None]).sum(1)
    return channels


def radiative_features(pc):
    channels=radiative_channels(pc)
    if getattr(pc,'ct_activation','relu')=='softplus':
        channels=torch.nn.functional.softplus(channels,beta=pc.ct_beta)
    else:channels=channels.clamp_min(0)
    return channels.mean(1,keepdim=True)


def attenuation(pc):
    if getattr(pc,'ct_activation','relu')=='softplus_sum':
        # Same point attributes; combine the existing branches in latent attenuation
        # space to avoid the multiplicative opacity/radiative scale ambiguity.
        latent=pc._opacity+radiative_channels(pc).mean(1,keepdim=True)
        return torch.nn.functional.softplus(latent,beta=pc.ct_beta)
    return pc.get_opacity * radiative_features(pc)


def render(camera, pc, debug=False, attenuation_scale=None):
    screen = torch.zeros_like(pc.get_xyz, requires_grad=True) + 0
    if screen.requires_grad:
        screen.retain_grad()
    RasterSettings,Rasterizer,_,_=operators(getattr(pc,'ct_backend','standard'))
    settings = RasterSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2) if camera.mode == 1 else 1.,
        tanfovy=math.tan(camera.FoVy/2) if camera.mode == 1 else 1.,
        scale_modifier=1., viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform, campos=camera.camera_center,
        prefiltered=False, mode=camera.mode, debug=debug)
    density=attenuation(pc)
    if attenuation_scale is not None:density=density*attenuation_scale
    image, radii = Rasterizer(raster_settings=settings)(
        means3D=pc.get_xyz, means2D=screen, opacities=density,
        scales=pc.get_scaling, rotations=pc.get_rotation)
    visible=radii>0
    if attenuation_scale is not None:visible=visible&(attenuation_scale.squeeze(1)>0)
    return dict(render=image, viewspace_points=screen, radii=radii,
                visibility_filter=visible)


def query(pc, center, nvoxel, svoxel, debug=False, center_grad=1.):
    _,_,VoxelSettings,Voxelizer=operators(getattr(pc,'ct_backend','standard'))
    settings = VoxelSettings(
        scale_modifier=1., nVoxel_x=int(nvoxel[0]), nVoxel_y=int(nvoxel[1]),
        nVoxel_z=int(nvoxel[2]), sVoxel_x=float(svoxel[0]),
        sVoxel_y=float(svoxel[1]), sVoxel_z=float(svoxel[2]),
        center_x=float(center[0]), center_y=float(center[1]),
        center_z=float(center[2]), prefiltered=False, debug=debug)
    means=pc.get_xyz if center_grad==1 else pc.get_xyz.detach()+center_grad*(pc.get_xyz-pc.get_xyz.detach())
    volume, radii = Voxelizer(voxel_settings=settings)(
        means3D=means, opacities=attenuation(pc),
        scales=pc.get_scaling, rotations=pc.get_rotation)
    return dict(vol=volume, radii=radii)
