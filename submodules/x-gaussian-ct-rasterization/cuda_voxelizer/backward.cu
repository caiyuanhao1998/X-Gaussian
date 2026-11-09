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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
static __device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

static __global__ void computeCov3DCUDA(int P,
	const float3* means,
	const int* radii_x, const int* radii_y, const int* radii_z,
	const float* cov3Ds,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float center_x, float center_y, float center_z,
	const float* dL_dconic3D,
	float3* dL_dmean3D,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii_x[idx] > 0) || !(radii_y[idx] > 0) || !(radii_z[idx] > 0))
		return;

	float dVoxel_x = sVoxel_x / (float)nVoxel_x;
	float dVoxel_y = sVoxel_y / (float)nVoxel_y;
	float dVoxel_z = sVoxel_z / (float)nVoxel_z;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;
	float3 mean = means[idx];

	float dL_dconic_a = dL_dconic3D[idx * 6 + 0];
	float dL_dconic_b = dL_dconic3D[idx * 6 + 1];
	float dL_dconic_c = dL_dconic3D[idx * 6 + 2];
	float dL_dconic_d = dL_dconic3D[idx * 6 + 3];
	float dL_dconic_e = dL_dconic3D[idx * 6 + 4];
	float dL_dconic_f = dL_dconic3D[idx * 6 + 5];

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 M = glm::mat3(
		1.0f / dVoxel_x, 0.0f, 0.0f,
		0.0f, 1.0f  / dVoxel_y, 0.0f,
		0.0f, 0.0f, 1.0f / dVoxel_z);
	glm::mat3 cov = glm::transpose(M) * glm::transpose(Vrk) * M;

	float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float denom = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;

	if (denom2inv != 0)
	{
		float denom_da = hatd * hatf - hate * hate;
		float denom_db = 2 * hatc * hate - 2 * hatf * hatb;
		float denom_dc = 2 * hatb * hate - 2 * hatd * hatc;
		float denom_dd = hata * hatf - hatc * hatc;
		float denom_de = 2 * hatb * hatc - 2 * hata * hate;
		float denom_df = hata * hatd - hatb * hatb;

		float ce_bf = hatc * hate - hatb * hatf;
		float be_cd = hatb * hate - hatc * hatd;
		float bc_ae = hatb * hatc - hata * hate;

		dL_dhata = denom2inv * (-denom_da*denom_da*dL_dconic_a - ce_bf*denom_da*dL_dconic_b - be_cd*denom_da*dL_dconic_c + (hatf*denom-denom_dd*denom_da)*dL_dconic_d + (-hate*denom-bc_ae*denom_da)*dL_dconic_e + (hatd*denom-denom_df*denom_da)*dL_dconic_f);
		dL_dhatb = denom2inv * (-denom_da*denom_db*dL_dconic_a + (-hatf*denom-ce_bf*denom_db)*dL_dconic_b + (hate*denom-be_cd*denom_db)*dL_dconic_c - denom_dd*denom_db*dL_dconic_d + (hatc*denom-bc_ae*denom_db)*dL_dconic_e + (-2*hatb*denom-denom_df*denom_db)*dL_dconic_f);
		dL_dhatc = denom2inv * (-denom_da*denom_dc*dL_dconic_a + (hate*denom-ce_bf*denom_dc)*dL_dconic_b + (-hatd*denom-be_cd*denom_dc)*dL_dconic_c + (-2*hatc*denom-denom_dd*denom_dc)*dL_dconic_d + (hatb*denom-bc_ae*denom_dc)*dL_dconic_e - denom_df*denom_dc*dL_dconic_f);
		dL_dhatd = denom2inv * ((hatf*denom-denom_da*denom_dd)*dL_dconic_a - ce_bf*denom_dd*dL_dconic_b +(-hatc*denom-be_cd*denom_dd)*dL_dconic_c - denom_dd*denom_dd*dL_dconic_d - bc_ae*denom_dd*dL_dconic_e + (hata*denom-denom_df*denom_dd)*dL_dconic_f);
		dL_dhate = denom2inv * ((-2*hate*denom-denom_da*denom_de)*dL_dconic_a + (hatc*denom-ce_bf*denom_de)*dL_dconic_b + (hatb*denom-be_cd*denom_de)*dL_dconic_c - denom_dd*denom_de*dL_dconic_d + (-hata*denom-bc_ae*denom_de)*dL_dconic_e + -denom_df*denom_de*dL_dconic_f);
		dL_dhatf = denom2inv * ((hatd*denom-denom_da*denom_df)*dL_dconic_a + (-hatb*denom-ce_bf*denom_df)*dL_dconic_b - be_cd*denom_df*dL_dconic_c + (hata*denom-denom_dd*denom_df)*dL_dconic_d - bc_ae*denom_df*dL_dconic_e - denom_df*denom_df*dL_dconic_f);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry
		// dL_da
		dL_dcov[6 * idx + 0] += M[0][0]*M[0][0]*dL_dhata + M[0][0]*M[1][0]*dL_dhatb + M[0][0]*M[2][0]*dL_dhatc + M[1][0]*M[1][0]*dL_dhatd + M[1][0]*M[2][0]*dL_dhate + M[2][0]*M[2][0]*dL_dhatf;
		// dL_dd
		dL_dcov[6 * idx + 3] += M[0][1]*M[0][1]*dL_dhata + M[0][1]*M[1][1]*dL_dhatb + M[0][1]*M[2][1]*dL_dhatc + M[1][1]*M[1][1]*dL_dhatd + M[1][1]*M[2][1]*dL_dhate + M[2][1]*M[2][1]*dL_dhatf;
		// dL_df
		dL_dcov[6 * idx + 5] += M[0][2]*M[0][2]*dL_dhata + M[0][2]*M[1][2]*dL_dhatb + M[0][2]*M[2][2]*dL_dhatc + M[1][2]*M[1][2]*dL_dhatd + M[1][2]*M[2][2]*dL_dhate + M[2][2]*M[2][2]*dL_dhatf;

		// dL_db
		dL_dcov[6 * idx + 1] += 2*M[0][0]*M[0][1]*dL_dhata + (M[0][1]*M[1][0]+M[0][0]*M[1][1])*dL_dhatb + (M[0][1]*M[2][0]+M[0][0]*M[2][1])*dL_dhatc + 2*M[1][0]*M[1][1]*dL_dhatd + (M[1][1]*M[2][0]+M[1][0]*M[2][1])*dL_dhate + 2*M[2][0]*M[2][1]*dL_dhatf;
		// dL_dc
		dL_dcov[6 * idx + 2] += 2*M[0][0]*M[0][2]*dL_dhata + (M[0][2]*M[1][0]+M[0][0]*M[1][2])*dL_dhatb + (M[0][2]*M[2][0]+M[0][0]*M[2][2])*dL_dhatc + 2*M[1][0]*M[1][2]*dL_dhatd + (M[1][2]*M[2][0]+M[1][0]*M[2][2])*dL_dhate + 2*M[2][0]*M[2][2]*dL_dhatf;
		// dL_de
		dL_dcov[6 * idx + 4] += 2*M[0][1]*M[0][2]*dL_dhata + (M[0][2]*M[1][1]+M[0][1]*M[1][2])*dL_dhatb + (M[0][2]*M[2][1]+M[0][1]*M[2][2])*dL_dhatc + 2*M[1][1]*M[1][2]*dL_dhatd + (M[1][2]*M[2][1]+M[1][1]*M[2][2])*dL_dhate + 2*M[2][1]*M[2][2]*dL_dhatf;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
}


template<int C>
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii_x, const int* radii_y, const int* radii_z,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float3* dL_dmean3D_norm,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii_x[idx] > 0) || !(radii_y[idx] > 0) || !(radii_z[idx] > 0))
		return;

	glm::vec3 dL_dmean;
	dL_dmean.x = dL_dmean3D_norm[idx].x;
	dL_dmean.y = dL_dmean3D_norm[idx].y;
	dL_dmean.z = dL_dmean3D_norm[idx].z;

	// printf("dL_dmean.x: %f\n", dL_dmean.x);
	// printf("dL_dmean.y: %f\n", dL_dmean.y);
	// printf("dL_dmean.z: %f\n", dL_dmean.z);

	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK3D_X * BLOCK3D_Y * BLOCK3D_Z)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float3* __restrict__ points_xyz_vol,
	const float* __restrict__ conic_opacity,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean3D_norm,
	float* __restrict__ dL_dconic3D,
	float* __restrict__ dL_dopacity)
{
	auto block = cg::this_thread_block();

	float dVoxel_x = sVoxel_x / (float)nVoxel_x;
	float dVoxel_y = sVoxel_y / (float)nVoxel_y;
	float dVoxel_z = sVoxel_z / (float)nVoxel_z;

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

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[voxel_id] : 0;

	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i *  nVoxel_x * nVoxel_y * nVoxel_z + voxel_id];

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	// World means are mapped to voxel coordinates by division by spacing.
	// The upstream multiplier used spacing and failed finite differences.
	const float ddelx_dx = 1.0f / dVoxel_x;
	const float ddely_dy = 1.0f / dVoxel_y;
	const float ddelz_dz = 1.0f / dVoxel_z;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK3D_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK3D_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
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

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK3D_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.

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

			if (power > 0.0f)
				continue;

			const float G = exp(power);

			// float alpha = min(1.0f, opa * G);
			float alpha = opa * G;
			if (alpha < 0.000001f)
				continue;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			// Since we are simple sum, dchannel_dalpha = 1.0
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += 1.0f * dL_dchannel;
			}

			// Helpful reusable temporary variables
			const float dL_dG = opa * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float gdz = G * d.z;
			const float dG_ddelx = - conic_a * gdx - conic_b * gdy - conic_c * gdz;
			const float dG_ddely = - conic_d * gdy - conic_b * gdx - conic_e * gdz;
			const float dG_ddelz = - conic_f * gdz - conic_c * gdx - conic_e * gdy;

			atomicAdd(&dL_dmean3D_norm[global_id].x, dL_dG * dG_ddelx * ddelx_dx); // ddelx_dx is used to compensate ndc2pix
			atomicAdd(&dL_dmean3D_norm[global_id].y, dL_dG * dG_ddely * ddely_dy); // ddelx_dx is used to compensate ndc2pix
			atomicAdd(&dL_dmean3D_norm[global_id].z, dL_dG * dG_ddelz * ddelz_dz); // ddelx_dx is used to compensate ndc2pix

			atomicAdd(&dL_dconic3D[global_id * 6 + 0], - 0.5 * gdx * d.x * dL_dG);  // conic_a
			atomicAdd(&dL_dconic3D[global_id * 6 + 1], - 1.0 * gdx * d.y * dL_dG);  // conic_b
			atomicAdd(&dL_dconic3D[global_id * 6 + 2], - 1.0 * gdx * d.z * dL_dG);  // conic_c
			atomicAdd(&dL_dconic3D[global_id * 6 + 3], - 0.5 * gdy * d.y * dL_dG);  // conic_d
			atomicAdd(&dL_dconic3D[global_id * 6 + 4], - 1.0 * gdy * d.z * dL_dG);  // conic_e
			atomicAdd(&dL_dconic3D[global_id * 6 + 5], - 0.5 * gdz * d.z * dL_dG);  // conic_f

			atomicAdd(&dL_dopacity[global_id], G * dL_dalpha);
		}
	}

}


void BACKWARD::preprocess(
	int P,
	const float3* means3D,
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
	glm::vec4* dL_drot)
{
	computeCov3DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii_x, radii_y, radii_z,
		cov3Ds,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		dL_dconic3D,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii_x, radii_y, radii_z,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		(float3*)dL_dmean3D_norm,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);

}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	float* dL_dopacity
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		means3D_norm,
		conic_opacity,
		n_contrib,
		dL_dpixels,
		dL_dmean3D_norm,
		dL_dconic3D,
		dL_dopacity
		);
}
