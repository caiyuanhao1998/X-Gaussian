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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_voxelizer/config.h"
#include "cuda_voxelizer/voxelizer.h"
#include <fstream>
#include <string>
#include <functional>
#include "utility.h"


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
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_volume = torch::full({nVoxel_x, nVoxel_y, nVoxel_z}, 0.0, float_opts);
	torch::Tensor radii_x = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor radii_y = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor radii_z = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if(P != 0)
	{
		rendered = CudaVoxelizer::Voxelizer::forward(
            geomFunc,
            binningFunc,
			imgFunc,
            P,
			nVoxel_x, nVoxel_y, nVoxel_z,
			sVoxel_x, sVoxel_y, sVoxel_z,
			center_x, center_y, center_z,
            means3D.contiguous().data<float>(),
            opacity.contiguous().data<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(),
            prefiltered,
            out_volume.contiguous().data<float>(),
            radii_x.contiguous().data<int>(),
			radii_y.contiguous().data<int>(),
			radii_z.contiguous().data<int>(),
            debug);
	}

	return std::make_tuple(rendered, out_volume, radii_x, radii_y, radii_z, geomBuffer, binningBuffer, imgBuffer);
}



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
	const bool debug)
{
	const int P = means3D.size(0);

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans3D_norm = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dconic3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	if(P != 0)
	{
		CudaVoxelizer::Voxelizer::backward(P, R,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		means3D.contiguous().data<float>(),
		scales.data_ptr<float>(),
		scale_modifier,
		rotations.data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),
		radii_x.contiguous().data<int>(),
		radii_y.contiguous().data<int>(),
		radii_z.contiguous().data<int>(),
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		dL_dout_color.contiguous().data<float>(),
		dL_dmeans3D_norm.contiguous().data<float>(),
		dL_dconic3D.contiguous().data<float>(),
		dL_dopacity.contiguous().data<float>(),
		dL_dmeans3D.contiguous().data<float>(),
		dL_dcov3D.contiguous().data<float>(),
		dL_dscales.contiguous().data<float>(),
		dL_drotations.contiguous().data<float>(),
		debug);
	}

	return std::make_tuple(dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations);
}
