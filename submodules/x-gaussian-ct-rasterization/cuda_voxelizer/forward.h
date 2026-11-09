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

#ifndef CUDA_VOXELIZER_FORWARD_H_INCLUDED
#define CUDA_VOXELIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* cov3D_precomp,
		const int nVoxel_x, int nVoxel_y, int nVoxel_z,
		const float sVoxel_x, float sVoxel_y, float sVoxel_z,
		const float center_x, float center_y, float center_z,
		int* radii_x, int* radii_y, int* radii_z,
		float3* means3D_norm,
		float* depths,
		float* cov3Ds,
		float* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered
		);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const int nVoxel_x, int nVoxel_y, int nVoxel_z,
		const float3* means3D_norm,
		const float* conic_opacity,
		uint32_t* n_contrib,
		float* out_volume);
}


#endif
