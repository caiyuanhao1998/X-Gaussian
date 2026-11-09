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

#include "voxelizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
static uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
static __global__ void duplicateWithKeys(
	int P,
	const float3* points_xyz_vol,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii_x,
	int* radii_y,
	int* radii_z,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii_x[idx] > 0 && radii_y[idx] > 0 && radii_z[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint3 cube_min, cube_max;
		float3 radii = { (float)radii_x[idx], (float)radii_y[idx], (float)radii_z[idx] };
		getCube(points_xyz_vol[idx], radii, cube_min, cube_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int z = cube_min.z; z < cube_max.z; z++)
		{
			for (int y = cube_min.y; y < cube_max.y; y++)
			{
				for (int x = cube_min.x; x < cube_max.x; x++)
				{
					uint64_t key = z * grid.y * grid.x + y * grid.x + x;
					key <<= 32;
					key |= *((uint32_t*)&depths[idx]);
					gaussian_keys_unsorted[off] = key;
					gaussian_values_unsorted[off] = idx;
					off++;
				}
			}
		}

	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
static __global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

CudaVoxelizer::GeometryState CudaVoxelizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.internal_radii_x, P, 128);
	obtain(chunk, geom.internal_radii_y, P, 128);
	obtain(chunk, geom.internal_radii_z, P, 128);
	obtain(chunk, geom.means3D_norm, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P * 7, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaVoxelizer::ImageState CudaVoxelizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}


CudaVoxelizer::BinningState CudaVoxelizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

int CudaVoxelizer::Voxelizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float center_x, float center_y, float center_z,
	const float* means3D,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const bool prefiltered,
	float* out_volume,
	int* radii_x,
	int* radii_y,
	int* radii_z,
	bool debug)
{

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii_x == nullptr)
	{
		radii_x = geomState.internal_radii_x;
	}
	if (radii_y == nullptr)
	{
		radii_y = geomState.internal_radii_y;
	}
	if (radii_z == nullptr)
	{
		radii_z = geomState.internal_radii_z;
	}

	dim3 tile_grid((nVoxel_x + BLOCK3D_X - 1) / BLOCK3D_X, (nVoxel_y + BLOCK3D_Y - 1) / BLOCK3D_Y, (nVoxel_z + BLOCK3D_Z - 1) / BLOCK3D_Z);
	dim3 block(BLOCK3D_X, BLOCK3D_Y, BLOCK3D_Z);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(nVoxel_x * nVoxel_y * nVoxel_z);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, nVoxel_x * nVoxel_y * nVoxel_z);


	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		cov3D_precomp,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		radii_x,
		radii_y,
		radii_z,
		geomState.means3D_norm,
		geomState.depths,
		geomState.cov3D,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means3D_norm,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii_x,
		radii_y,
		radii_z,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y * tile_grid.z);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * tile_grid.z * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		nVoxel_x, nVoxel_y, nVoxel_z,
		geomState.means3D_norm,
		geomState.conic_opacity,
		imgState.n_contrib,
		out_volume), debug)


    return num_rendered;
}


// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaVoxelizer::Voxelizer::backward(
	const int P, int R,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float center_x, float center_y, float center_z,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const int* radii_x,
	const int* radii_y,
	const int* radii_z,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean3D_norm,
	float* dL_dconic3D,
	float* dL_dopacity,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, nVoxel_x * nVoxel_y * nVoxel_z);

	if (radii_x == nullptr)
	{
		radii_x = geomState.internal_radii_x;
	}
	if (radii_y == nullptr)
	{
		radii_y = geomState.internal_radii_y;
	}
	if (radii_z == nullptr)
	{
		radii_z = geomState.internal_radii_z;
	}

	dim3 tile_grid((nVoxel_x + BLOCK3D_X - 1) / BLOCK3D_X, (nVoxel_y + BLOCK3D_Y - 1) / BLOCK3D_Y, (nVoxel_z + BLOCK3D_Z - 1) / BLOCK3D_Z);
	dim3 block(BLOCK3D_X, BLOCK3D_Y, BLOCK3D_Z);

	// Compute loss gradients w.r.t. 3D mean position, conic matrix,
	// opacity.
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		geomState.means3D_norm,
		geomState.conic_opacity,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean3D_norm,
		dL_dconic3D,
		dL_dopacity), debug)

	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P,
		(float3*)means3D,
		radii_x,
		radii_y,
		radii_z,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		(float3*)dL_dmean3D_norm,
		dL_dconic3D,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
