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
#include <stdio.h> // for debug
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
static __device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];  // a
	cov3D[1] = Sigma[0][1];  // b
	cov3D[2] = Sigma[0][2];  // c
	cov3D[3] = Sigma[1][1];  // d
	cov3D[4] = Sigma[1][2];  // e
	cov3D[5] = Sigma[2][2];  // f
}


template<int C>
__global__ void preprocessCUDA(int P,
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
	float3* points_xyz_vol,
	float* depths,
	float* cov3Ds,
	float* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered
	)
{
    auto idx = cg::this_grid().thread_rank(); // idx
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii_x[idx] = 0;
	radii_y[idx] = 0;
	radii_z[idx] = 0;
	tiles_touched[idx] = 0;

	float dVoxel_x = sVoxel_x / (float)nVoxel_x;
	float dVoxel_y = sVoxel_y / (float)nVoxel_y;
	float dVoxel_z = sVoxel_z / (float)nVoxel_z;

	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Transfer to voxel space
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 M = glm::mat3(
		1.f / dVoxel_x, 0.0f, 0.0f,
		0.0f, 1.f  / dVoxel_y, 0.0f,
		0.0f, 0.0f, 1.f / dVoxel_z);
	glm::mat3 cov = glm::transpose(M) * glm::transpose(Vrk) * M;

	float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float det = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float inv_a = (hatd * hatf - hate * hate) * det_inv;
	float inv_b = (hatc * hate - hatb * hatf) * det_inv;
	float inv_c = (hatb * hate - hatc * hatd) * det_inv;
	float inv_d = (hata * hatf - hatc * hatc) * det_inv;
	float inv_e = (hatb * hatc - hata * hate) * det_inv;
	float inv_f = (hata * hatd - hatb * hatb) * det_inv;

	glm::vec3 scale = scales[idx];
	float max_scale = max(max(scale.x, scale.y), scale.z);
	float3 my_radius = {
		ceil((3.f * max_scale) / dVoxel_x),
		ceil((3.f * max_scale) / dVoxel_y),
		ceil((3.f * max_scale) / dVoxel_z)
	};

	float3 point_vol = {(p_orig.x - center_x + sVoxel_x / 2) / dVoxel_x,
						(p_orig.y - center_y + sVoxel_y / 2) / dVoxel_y,
						(p_orig.z - center_z + sVoxel_z / 2) / dVoxel_z};


    if (point_vol.x + my_radius.x< 0 || point_vol.y + my_radius.y < 0 || point_vol.z + my_radius.z < 0 || point_vol.x - my_radius.x > (float)nVoxel_x || point_vol.y - my_radius.y > (float)nVoxel_y || point_vol.z - my_radius.z > (float)nVoxel_z)
	{
		return;
	}

	uint3 cube_min, cube_max;
	getCube(point_vol, my_radius, cube_min, cube_max, grid);


	if ((cube_max.x - cube_min.x) * (cube_max.y - cube_min.y) * (cube_max.z - cube_min.z) == 0)
		return;

	radii_x[idx] = my_radius.x;
	radii_y[idx] = my_radius.y;
	radii_z[idx] = my_radius.z;

	// printf("radii_x: %d, radii_y: %d, radii_z: %d\n", radii_x[idx], radii_y[idx], radii_z[idx]);

	tiles_touched[idx] = (cube_max.z - cube_min.z) * (cube_max.y - cube_min.y) * (cube_max.x - cube_min.x);
	depths[idx] = p_orig.z;  // just give a value
	points_xyz_vol[idx] = point_vol;
	conic_opacity[idx * 7 + 0] = inv_a;
	conic_opacity[idx * 7 + 1] = inv_b;
	conic_opacity[idx * 7 + 2] = inv_c;
	conic_opacity[idx * 7 + 3] = inv_d;
	conic_opacity[idx * 7 + 4] = inv_e;
	conic_opacity[idx * 7 + 5] = inv_f;
	conic_opacity[idx * 7 + 6] = opacities[idx];
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK3D_X * BLOCK3D_Y * BLOCK3D_Z)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float3* __restrict__ points_xyz_vol,
	const float* __restrict__ conic_opacity,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_volume
	)
{

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	uint32_t horizontal_blocks1 = (nVoxel_x + BLOCK3D_X - 1) / BLOCK3D_X;
	uint32_t horizontal_blocks2 = (nVoxel_y + BLOCK3D_Y - 1) / BLOCK3D_Y;
	uint3 voxel_min = { block.group_index().x * BLOCK3D_X, block.group_index().y * BLOCK3D_Y,  block.group_index().z * BLOCK3D_Z};
	uint3 voxel_max = { min(voxel_min.x + BLOCK3D_X, nVoxel_x), min(voxel_min.y + BLOCK3D_Y , nVoxel_y),  min(voxel_min.z + BLOCK3D_Z , nVoxel_z)};
	uint3 voxel = { voxel_min.x + block.thread_index().x, voxel_min.y + block.thread_index().y, voxel_min.z + block.thread_index().z};
	uint32_t voxel_id = nVoxel_z * nVoxel_y * voxel.x + nVoxel_z * voxel.y + voxel.z;
	// add 0.5 because we donnot count offset previously, like gs code in ndc2pixel()
	float3 voxelf = { (float)voxel.x + 0.5f, (float)voxel.y + 0.5f, (float)voxel.z + 0.5f};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = voxel.x < nVoxel_x && voxel.y < nVoxel_y && voxel.z < nVoxel_z;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().z * horizontal_blocks2 * horizontal_blocks1 + block.group_index().y * horizontal_blocks1 + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK3D_SIZE - 1) / BLOCK3D_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK3D_SIZE];
	__shared__ float3 collected_xyz[BLOCK3D_SIZE];
	__shared__ float collected_conic_a[BLOCK3D_SIZE];
	__shared__ float collected_conic_b[BLOCK3D_SIZE];
	__shared__ float collected_conic_c[BLOCK3D_SIZE];
	__shared__ float collected_conic_d[BLOCK3D_SIZE];
	__shared__ float collected_conic_e[BLOCK3D_SIZE];
	__shared__ float collected_conic_f[BLOCK3D_SIZE];
	__shared__ float collected_o[BLOCK3D_SIZE];

	// Initialize helper variables
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK3D_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK3D_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK3D_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = points_xyz_vol[coll_id];
			collected_conic_a[block.thread_rank()] = conic_opacity[coll_id * 7 + 0];
			collected_conic_b[block.thread_rank()] = conic_opacity[coll_id * 7 + 1];
			collected_conic_c[block.thread_rank()] = conic_opacity[coll_id * 7 + 2];
			collected_conic_d[block.thread_rank()] = conic_opacity[coll_id * 7 + 3];
			collected_conic_e[block.thread_rank()] = conic_opacity[coll_id * 7 + 4];
			collected_conic_f[block.thread_rank()] = conic_opacity[coll_id * 7 + 5];
			collected_o[block.thread_rank()] = conic_opacity[coll_id * 7 + 6];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK3D_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			float3 xyz = collected_xyz[j];
			float3 d = { xyz.x - voxelf.x, xyz.y - voxelf.y, xyz.z - voxelf.z };
			float conic_a = collected_conic_a[j];
			float conic_b = collected_conic_b[j];
			float conic_c = collected_conic_c[j];
			float conic_d = collected_conic_d[j];
			float conic_e = collected_conic_e[j];
			float conic_f = collected_conic_f[j];
			float opa = collected_o[j];

			float power = - 0.5 * (conic_a * d.x * d.x + conic_d * d.y * d.y + conic_f * d.z * d.z) - conic_b * d.x * d.y - conic_c * d.x * d.z - conic_e * d.y * d.z;

			// printf("conic_a: %f\n", conic_a);
			// printf("conic_b: %f\n", conic_b);
			// printf("conic_c: %f\n", conic_c);
			// printf("conic_d: %f\n", conic_d);
			// printf("conic_e: %f\n", conic_e);
			// printf("conic_f: %f\n", conic_f);
			// printf("opa: %f\n", opa);
			// printf("xyz.x: %f, xyz.y: %f, xyz.z: %f\n", xyz.x, xyz.y, xyz.z);
			// printf("voxelf.x: %f, voxelf.y: %f, voxelf.z: %f\n", voxelf.x, voxelf.y, voxelf.z);
			// printf("d.x: %f, d.y: %f, d.z: %f\n", d.x, d.y, d.z);
			// printf("power: %f\n", power);

			if (power > 0.0f)
				continue;

			// float alpha = min(1.0f, opa * exp(power));
			float alpha = opa * exp(power);
			if (alpha < 0.000001f)
				continue;

			// Simply add all alphas
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += alpha;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}

	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		n_contrib[voxel_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_volume[ch * nVoxel_x * nVoxel_y * nVoxel_z + voxel_id] = C[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float3* means3D_norm,
	const float* conic_opacity,
	uint32_t* n_contrib,
	float* out_volume)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		nVoxel_x, nVoxel_y, nVoxel_z,
		means3D_norm,
		conic_opacity,
		n_contrib,
		out_volume
		);

}


void FORWARD::preprocess(int P,
	const float* means3D,
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
	)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		cov3D_precomp,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		radii_x, radii_y, radii_z,
		means3D_norm,
		depths,
		cov3Ds,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
