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

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
//! We dont need it.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	const float* dL_dmus,
	float3* dL_dmeans,
	float* dL_dcov,
	const int mode)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float dL_dmu = dL_dmus[idx];
	float3 t = transformPoint4x3(mean, view_matrix);

	glm::mat3 J;
	float x_grad_mul, y_grad_mul;
	if (mode == 0){  //! parallel beam
		const float limx = 1.3f;
		const float limy = 1.3f;
		t.x = min(limx, max(-limx, t.x));
		t.y = min(limx, max(-limx, t.y));

		x_grad_mul = t.x < -limx || t.x > limx ? 0 : 1;
		y_grad_mul = t.y < -limy || t.y > limy ? 0 : 1;

		J = glm::mat3(
		h_x, 0.0f, 0.0f,
		0.0f, h_y, 0.0f,
		0, 0, 1.0f);
	}
	else  //! cone beam
	{
		const float limx = 1.3f * tan_fovx;
		const float limy = 1.3f * tan_fovy;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;

		x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
		y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

		const float l = sqrt(t.x * t.x +  t.y * t.y + t.z * t.z);
		J = glm::mat3(
			h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
			0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
			t.x / l, t.y / l, t.z / l);  //! We need last row for computation.
	}

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 M = W * J;

	glm::mat3 cov = glm::transpose(M) * glm::transpose(Vrk) * M;

	// Use helper variables for 2D covariance entries. More compact.
	float hata = cov[0][0] += 0.0f;
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1] += 0.0f;
	float hate = cov[1][2];
	float hatf = cov[2][2];

	float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;
	float denom = hata * hatd - hatb * hatb;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
	float diamond = hata * hatd - hatb * hatb;

	//! mu part gradient
	float circ = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float mu_square = 2 * M_PI * circ / diamond;
	float mu = 0.0f;
	if (mu_square > 0.0f){
		mu = sqrt(2 * M_PI * circ / diamond);
	}
	float pi_mu= M_PI / (mu + 0.0000001f);
	float circ_diamond = circ / diamond;

	if (denom2inv != 0.0f && mu != 0.0f)
	{
		// exp(*) part gradient
		dL_dhata = denom2inv * (-hatd * hatd * dL_dconic.x + hatb * hatd * dL_dconic.y + (denom - hata * hatd) * dL_dconic.z);  // We remove 2 here because in render we do not *0.5
		dL_dhatd = denom2inv * (-hata * hata * dL_dconic.z + hata * hatb * dL_dconic.y + (denom - hata * hatd) * dL_dconic.x);  // We remove 2 here because in render we do not *0.5
		dL_dhatb = denom2inv * (2 * hatb * hatd * dL_dconic.x - (denom + 2 * hatb * hatb) * dL_dconic.y + 2 * hata * hatb * dL_dconic.z);

		dL_dhata += pi_mu * ((hatd * hatf - hate * hate) / diamond -  hatd * circ_diamond / diamond) * dL_dmu;
		dL_dhatb += pi_mu * ((2 * hatc * hate - 2 * hatf * hatb) / diamond + 2 * hatb * circ_diamond / diamond) * dL_dmu;
		dL_dhatc += pi_mu * ((2 * hatb * hate - 2 * hatd * hatc) / diamond) * dL_dmu;
		dL_dhatd += pi_mu * ((hata * hatf - hatc * hatc) / diamond -  hata *circ_diamond / diamond) * dL_dmu;
		dL_dhate += pi_mu * ((2 * hatb * hatc - 2 * hata * hate) / diamond) * dL_dmu;
		dL_dhatf += pi_mu * ((hata * hatd - hatb * hatb) / diamond) * dL_dmu;

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

	if (mode == 1){
		// Gradients of loss w.r.t. M
		// cov2D = transpose(M) * transpose(Vrk) * M;

		float a = cov3D[0];
		float b = cov3D[1];
		float c = cov3D[2];
		float d = cov3D[3];
		float e = cov3D[4];
		float f = cov3D[5];

		float dL_dM00 = 2*(M[0][0]*a+M[0][1]*b + M[0][2]*c)*dL_dhata + (M[1][0]*a+M[1][1]*b+M[1][2]*c)*dL_dhatb + (M[2][0]*a+M[2][1]*b+M[2][2]*c)*dL_dhatc;
		float dL_dM01 = 2*(M[0][0]*b+M[0][1]*d + M[0][2]*e)*dL_dhata + (M[1][0]*b+M[1][1]*d+M[1][2]*e)*dL_dhatb + (M[2][0]*b+M[2][1]*d+M[2][2]*e)*dL_dhatc;
		float dL_dM02 = 2*(M[0][0]*c+M[0][1]*e + M[0][2]*f)*dL_dhata + (M[1][0]*c+M[1][1]*e+M[1][2]*f)*dL_dhatb + (M[2][0]*c+M[2][1]*e+M[2][2]*f)*dL_dhatc;

		float dL_dM10 = (M[0][0]*a+M[0][1]*b+M[0][2]*c)*dL_dhatb + 2*(M[1][0]*a+M[1][1]*b+M[1][2]*c)*dL_dhatd + (M[2][0]*a+M[2][1]*b+M[2][2]*c)*dL_dhate;
		float dL_dM11 = (M[0][0]*b+M[0][1]*d+M[0][2]*e)*dL_dhatb + 2*(M[1][0]*b+M[1][1]*d+M[1][2]*e)*dL_dhatd + (M[2][0]*b+M[2][1]*d+M[2][2]*e)*dL_dhate;
		float dL_dM12 = (M[0][0]*c+M[0][1]*e+M[0][2]*f)*dL_dhatb + 2*(M[1][0]*c+M[1][1]*e+M[1][2]*f)*dL_dhatd + (M[2][0]*c+M[2][1]*e+M[2][2]*f)*dL_dhate;

		float dL_dM20 = (M[0][0]*a+M[0][1]*b+M[0][2]*c)*dL_dhatc + (M[1][0]*a+M[1][1]*b+M[1][2]*c)*dL_dhate + 2*(M[2][0]*a+M[2][1]*b+M[2][2]*c)*dL_dhatf;
		float dL_dM21 = (M[0][0]*b+M[0][1]*d+M[0][2]*e)*dL_dhatc + (M[1][0]*b+M[1][1]*d+M[1][2]*e)*dL_dhate + 2*(M[2][0]*b+M[2][1]*d+M[2][2]*e)*dL_dhatf;
		float dL_dM22 = (M[0][0]*c+M[0][1]*e+M[0][2]*f)*dL_dhatc + (M[1][0]*c+M[1][1]*e+M[1][2]*f)*dL_dhate + 2*(M[2][0]*c+M[2][1]*e+M[2][2]*f)*dL_dhatf;

		float dL_dJ00 = W[0][0]*dL_dM00 + W[0][1]*dL_dM01 + W[0][2]*dL_dM02;
		float dL_dJ02 = W[2][0]*dL_dM00 + W[2][1]*dL_dM01 + W[2][2]*dL_dM02;
		float dL_dJ11 = W[1][0]*dL_dM10 + W[1][1]*dL_dM11 + W[1][2]*dL_dM12;
		float dL_dJ12 = W[2][0]*dL_dM10 + W[2][1]*dL_dM11 + W[2][2]*dL_dM12;
		float dL_dJ20 = W[0][0]*dL_dM20 + W[0][1]*dL_dM21 + W[0][2]*dL_dM22;
		float dL_dJ21 = W[1][0]*dL_dM20 + W[1][1]*dL_dM21 + W[1][2]*dL_dM22;
		float dL_dJ22 = W[2][0]*dL_dM20 + W[2][1]*dL_dM21 + W[2][2]*dL_dM22;

		float tx = t.x;
		float ty = t.y;
		float tz = t.z;
		float inv_tz = 1.f / tz;
		float inv_tz2 = inv_tz * inv_tz;
		float inv_tz3 = inv_tz2 * inv_tz;
		float circledcirc = sqrt(tx * tx + ty * ty + tz * tz);
		float inv_circledcirc3 = 1 / (circledcirc * circledcirc * circledcirc);
		float dL_dtx = x_grad_mul * (-h_x*inv_tz2*dL_dJ02 + (1/circledcirc - tx*tx*inv_circledcirc3)*dL_dJ20 - tx*ty*inv_circledcirc3*dL_dJ21 - tx*tz*inv_circledcirc3*dL_dJ22);
		float dL_dty = y_grad_mul * (-h_y*inv_tz2*dL_dJ12 - tx*ty*inv_circledcirc3*dL_dJ20 + (1/circledcirc - ty*ty*inv_circledcirc3)*dL_dJ21 - ty*tz*inv_circledcirc3*dL_dJ22);
		float dL_dtz = -h_x*inv_tz2*dL_dJ00 + 2*h_x*tx*inv_tz3*dL_dJ02 - h_y*inv_tz2*dL_dJ11 + 2*h_y*ty*inv_tz3*dL_dJ12 - tx*tz*inv_circledcirc3*dL_dJ20 - ty*tz*inv_circledcirc3*dL_dJ21 + (1/circledcirc-tz*tz*inv_circledcirc3)*dL_dJ22;

		float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

		// Gradients of loss w.r.t. Gaussian means, but only the portion
		// that is caused because the mean affects the covariance matrix.
		// Additional mean gradient is accumulated in BACKWARD::preprocess.
		dL_dmeans[idx] = dL_dmean;

	}
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
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

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ mus,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dmu)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_mu[BLOCK_SIZE];

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_mu[block.thread_rank()] = mus[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			float mu = collected_mu[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);

			// const float alpha = min(1.0f, con_o.w * mu * G);
			const float alpha = con_o.w * mu * G;
			if (alpha <0.00001f)
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
				dL_dalpha += 1.f * dL_dchannel;
			}

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * mu * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx); // ddelx_dx is used to compensate ndc2pix
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy); // ddelx_dx is used to compensate ndc2pix

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -1.0f * gdx * d.y * dL_dG); // gs code is 0.5f, which is futher compensated in computecov2D
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), mu * G * dL_dalpha);
			atomicAdd(&(dL_dmu[global_id]), con_o.w * G * dL_dalpha);
		}
	}
}


void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dmu,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const int mode)
{
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		dL_dmu,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		mode);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float* mus,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dmu
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		mus,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dmu
		);
}
