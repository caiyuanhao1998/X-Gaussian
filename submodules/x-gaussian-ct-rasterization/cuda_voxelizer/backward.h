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

#ifndef CUDA_VOXELIZER_BACKWARD_H_INCLUDED
#define CUDA_VOXELIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const int nVoxel_x, int nVoxel_y, int nVoxel_z,
		const float sVoxel_x, float sVoxel_y, float sVoxel_z,
		const float3* means3D_norm,
		const float* conic_opacity,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		float3* dL_dmean3D_norm,
		float* dL_dconic3D,
		float* dL_dopacity);

	void preprocess(
		int P,
		const float3* means,
		const int* radii_x, const int* radii_y, const int* radii_z,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const int nVoxel_x, int nVoxel_y, int nVoxel_z,
		const float sVoxel_x, float sVoxel_y, float sVoxel_z,
		const float center_x, float center_y, float center_z,
		const float3* dL_dmean3D_norm,
		const float* dL_dconic3D,
		glm::vec3* dL_dmean3D,
		float* dL_dcov3D,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}


#endif
