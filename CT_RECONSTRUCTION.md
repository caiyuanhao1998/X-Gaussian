# Direct CT reconstruction with X-Gaussian

The recommended recipe preserves `GaussianModel_Xray`, its six optimizer groups,
59 learned scalars per point, and original 62-column PLY layout. The original 2D
`train.py` and renderer remain available. The [README testing section](README.md#3d-reconstruction-testing)
contains the reconstruction equations and runnable training/export commands.

## Recommended implementation

- `config/ct_teapot.yaml`: the measured product-head, original exp/log-scale recipe.
- `xct/field.py`: original X constant-weight radiative decoder, positive Softplus
  channels, sigmoid opacity, additive projection, and differentiable volume queries.
- `xct/refinement.py`: exact colocated clone invariance and integrated-mass-preserving
  splits, including the actual child scales after the scale floor.
- `train_ct.py`: 30,000 projection-training steps with local 32-cubed TV. TV gradients
  to centers are blocked (`tv_center_grad: 0`); projection gradients remain active.
- `finetune_ct_rays.py`: 1,000 fixed-topology steps. Query the original 256-cubed
  Gaussian field and integrate 8,192 original-detector rays per step using exact
  path lengths through the piecewise-constant voxel discretization. Ray weight is 2.
- `export_ct.py`: PLY-only volume export, without data, FDK or CT truth at inference.
- `evaluate_ct_model.py` and `verify_ct_export.py`: frozen-model metrics and an
  independent copied-PLY export/schema/loss check.

The stage-1 loss is

$$
\mathcal L_{\mathrm{base}}=0.8\,\mathcal L_1(P_\theta,P)
+0.2\,[1-\operatorname{SSIM}(P_\theta,P)]
+0.05\,\operatorname{TV}(V_\theta).
$$

Stage 2 additionally adds

$$
2\,\mathbb E_{u\in\mathcal R}\left|
\sum_{\mathbf j}L_{u\mathbf j}V_\theta[\mathbf j]-P(u)\right|.
$$

The TV term updates attenuation and covariance, but has a stopped gradient with
respect to centers. The native projection and Siddon terms both train centers.
Voxels are always computed from Gaussian attributes; no separate learned voxel
parameters or per-Gaussian density field are introduced. No CT truth enters either
loss. The original 48 radiative coefficients remain trainable, but their
view-independent weighted combination has redundant degrees of freedom; retaining
these coefficients is not a claim of additional independent CT observables.

## Installation and data

Use `environment_ct.yml`, CUDA 11.8 with `nvcc`, and a compatible host compiler.
`bash install_ct.sh` builds `simple-knn` and `x_gaussian_ct_rasterization` using the
active Python/PyTorch environment. `CT_PYTHON` overrides Python selection; a local
`.venv_ct/bin/python`, when present, is selected automatically. Set `CUDA_HOME`,
`CC`, and `CXX` if your CUDA toolkit/compiler is not on the default search path.

The data adapter supports cone-beam NeRF-format R2 datasets with `meta_data.json`.
It normalizes the largest scanner extent to 2, uses xyz voxel ordering, and never
resizes the supplied projections or volume. The official FDK point file
`init_<dataset-directory-name>.npy` must contain normalized xyz plus density.
Alternatively pass a training-only initialization through `train_ct.py --init_path`.
`prepare_ct_trainval.py` can create a 40-view or 50-view FDK initialization, but
requires the separately installed [TIGRE](https://github.com/CERN/TIGRE) package.
TIGRE is not needed when the official initialization is already available.

Use `DATA_PATH`, `BASE_PATH`, and `OUTPUT_PATH` with `run_ct_precision.sh` to store
large inputs and outputs outside the checkout. Existing datasets, environments,
model files and logs are not included in this source release. A CT-trained PLY must
be generated using the training recipe or supplied separately; the original 2D
paper's pretrained checkpoint is not a quantitatively calibrated CT model.

## Export conventions

`export_ct.py` writes an xyz-ordered `.npy` array sampled at voxel centers, plus a
JSON sidecar containing resolution, extent, center and spacing. The default query
box is the normalized scanner cube `[-1,1]^3`, not millimeter coordinates. The PLY
global `xct_field` comment records activation, beta, degree and backend. This metadata
adds no per-point columns and lets an isolated PLY be decoded without its run config.
The recommended model uses `activation=softplus` and `backend=standard`.

Historical CT heads can be decoded when their field metadata/config is available;
only the product-head standard-backend recipe is the recommendation. The local
experimental precision package is not required or bundled with this recipe.

## Measured protocol and limitations

The test case is the original R2 synthetic `2_teapot_cone`: 512 x 512 projections,
50 training / 100 official test views, 256-cubed volume, 50,000 FDK initial points,
seed 0. R2 is the previously trained, unchanged 30,000-step checkpoint. X uses
30,000 base steps plus 1,000 ray steps; the latter also add full-volume query and ray
integration work, so this is not an equal-compute comparison.

| Configuration | Steps | 2D PSNR | Raw 2D PSNR | 3D PSNR |
|---|---:|---:|---:|---:|
| Previous X CT final checkpoint | 31,000 | 54.2289 | 64.4766 | 46.3974 |
| TV center isolation, no ray stage | 30,000 | 54.0217 | 64.2134 | 47.0266 |
| **Recommended: TV center isolation + rays** | **31,000** | **53.7233** | **64.0590** | **47.4791** |
| Existing R2 baseline | 30,000 | 54.0160 | 64.2410 | 47.2091 |

The recommended X checkpoint has 51,359 points, 2D SSIM 0.988825 and 3D SSIM
0.990997; R2 has 50,029 points and SSIM 0.988537 / 0.990321. The small 3D advantage
has a native 2D PSNR cost. Full numeric results are in `results/ct_teapot.json`.

Initial screening used 40 training / 10 held-out training views and FDK rebuilt from
only those 40 training views. Ray settings were screened by held-out voxel-ray PSNR
with at most 0.2 dB native-projection PSNR loss against matched ordinary continuation.
Final full-data CT diagnostics informed the chosen source profile, so the final
recommendation is **post-hoc, adaptive engineering on a single case**. It is not an
unbiased held-out-selected superiority result or evidence of cross-case superiority.
All reported checkpoints are fixed training endpoints; no intermediate test-curve
maximum is selected. No GT-fitted gain, offset, smoothing, registration or clipping
is applied. The 2D official metric separately max-normalizes prediction/reference;
raw-intensity PSNR uses fixed range 1. 3D PSNR uses fixed range 1.

A conditioned sum-head alternative reached 47.5022 dB 3D / 53.6493 dB 2D, but changed
opacity's CT interpretation and offered only 0.0231 dB more 3D PSNR. The product head
was retained to preserve more X behavior. Increased initial point count, bounded
scale optimization, dropout and reduced CUDA cutoffs did not improve the selected
tradeoff and are not part of the recommended settings.

## Numerical validation

The default CUDA checks do not require a dataset or an R2 installation:

```shell
CUDA_VISIBLE_DEVICES=0 python tests/validate_ct.py
CUDA_VISIBLE_DEVICES=0 python tests/validate_ct_stable.py
CUDA_VISIBLE_DEVICES=0 python tests/validate_ct_conditioning.py
CUDA_VISIBLE_DEVICES=0 python tests/validate_ct_rays.py
```

They cover analytic Gaussian values and central-ray integrals, finite differences
for all six original parameter groups in projection and volume paths, positive tail
gradients, clone/split mass conservation, isolated-PLY metadata/schema, stopped TV
center gradients, and exact one-voxel ray lengths/adjoints. Conditioning tests also
cover optional historical optimizer coordinates; they are not required by the
recommended training settings.

To additionally compare actual camera/data inputs with an R2 checkout:

```shell
python tests/validate_ct.py --source_path /path/to/2_teapot_cone --r2_repo /path/to/r2_gaussian
```

To independently re-evaluate both frozen models, use the optional comparison tool
in an environment containing R2's CUDA packages:

```shell
python compare_ct.py --model_path output_ct/teapot_ct_recommended --iteration 1000 \
  --baseline /path/to/r2_output --r2_repo /path/to/r2_gaussian
```

The comparison writes full metrics, projection/slice/error figures and a loss plot.
The published figures in `fig/ct_teapot_*.png` come from the measured checkpoint.

## CUDA attribution and approximation

The separately named rasterizer/voxelizer is adapted from R2-Gaussian commit
`f2579bfddd9aac009cb797c8503bef8119bbd022`, retaining upstream licenses. The inherited
voxel-mean backward chain is corrected to multiply by inverse voxel spacing. The
original X renderer package is not replaced. See the backend's `UPSTREAM.md`.

Native projection uses the R2 local perspective ray-space approximation and finite
support/tile culling; it is not exact infinite-support cone-ray integration. The
independent voxel Siddon implementation integrates the chosen discretized field,
but is not claimed to be bitwise identical to TIGRE. Lowering CUDA cutoffs alone was
not an accuracy improvement in this case and is not enabled in the recommendation.
