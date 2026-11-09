/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelizeGaussiansCUDA(
	const torch::Tensor& means3D,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const int nVoxel_x,
    const int nVoxel_y,
    const int nVoxel_z,
    const float sVoxel_x,
    const float sVoxel_y,
    const float sVoxel_z,
    const float center_x,
    const float center_y,
    const float center_z,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelizeGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii_x,
	const torch::Tensor& radii_y,
	const torch::Tensor& radii_z,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int nVoxel_x,
    const int nVoxel_y,
    const int nVoxel_z,
    const float sVoxel_x,
    const float sVoxel_y,
    const float sVoxel_z,
    const float center_x,
    const float center_y,
    const float center_z,
	const bool debug);
