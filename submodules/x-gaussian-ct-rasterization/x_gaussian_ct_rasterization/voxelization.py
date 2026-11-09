#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


class GaussianVoxelizationSettings(NamedTuple):
    scale_modifier: float
    nVoxel_x: int
    nVoxel_y: int
    nVoxel_z: int
    sVoxel_x: float
    sVoxel_y: float
    sVoxel_z: float
    center_x: float
    center_y: float
    center_z: float
    prefiltered: bool
    debug: bool


def voxelize_gaussians(
    means3D,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    voxel_settings: GaussianVoxelizationSettings,
):
    return _VoxelizeGaussians.apply(
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        voxel_settings,
    )


class _VoxelizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        voxel_settings: GaussianVoxelizationSettings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            voxel_settings.scale_modifier,
            cov3Ds_precomp,
            voxel_settings.nVoxel_x,
            voxel_settings.nVoxel_y,
            voxel_settings.nVoxel_z,
            voxel_settings.sVoxel_x,
            voxel_settings.sVoxel_y,
            voxel_settings.sVoxel_z,
            voxel_settings.center_x,
            voxel_settings.center_y,
            voxel_settings.center_z,
            voxel_settings.prefiltered,
            voxel_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if voxel_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    fields,
                    radii_x,
                    radii_y,
                    radii_z,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                ) = _C.voxelize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                fields,
                radii_x,
                radii_y,
                radii_z,
                geomBuffer,
                binningBuffer,
                imgBuffer,
            ) = _C.voxelize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.voxel_settings = voxel_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii_x,
            radii_y,
            radii_z,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        radii = (radii_x, radii_y, radii_z)

        return fields, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        voxel_settings = ctx.voxel_settings
        (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii_x,
            radii_y,
            radii_z,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            means3D,
            radii_x,
            radii_y,
            radii_z,
            scales,
            rotations,
            voxel_settings.scale_modifier,
            cov3Ds_precomp,
            grad_out_color,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            voxel_settings.nVoxel_x,
            voxel_settings.nVoxel_y,
            voxel_settings.nVoxel_z,
            voxel_settings.sVoxel_x,
            voxel_settings.sVoxel_y,
            voxel_settings.sVoxel_z,
            voxel_settings.center_x,
            voxel_settings.center_y,
            voxel_settings.center_z,
            voxel_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if voxel_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_scales,
                    grad_rotations,
                ) = _C.voxelize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_scales,
                grad_rotations,
            ) = _C.voxelize_gaussians_backward(*args)
        grads = (
            grad_means3D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianVoxelizer(nn.Module):
    def __init__(self, voxel_settings):
        super().__init__()
        self.voxel_settings = voxel_settings

    def forward(
        self,
        means3D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):

        voxel_settings = self.voxel_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return voxelize_gaussians(
            means3D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            voxel_settings,
        )
