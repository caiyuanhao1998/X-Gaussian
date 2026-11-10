&nbsp;

<div align="center">

<p align="center"> <img src="3d_demo/logo.png" width="110px"> </p>

[![arXiv](https://img.shields.io/badge/arXiv-2403.04116-b31b1b)](https://arxiv.org/abs/2403.04116)
[![知乎](https://img.shields.io/badge/知乎-中文解读-0084ff)](https://zhuanlan.zhihu.com/p/717744222)
[![Video](https://img.shields.io/badge/YouTube-Video-ff0000?logo=youtube)](https://www.youtube.com/watch?v=v6FESb3SkJg&t=28s)
[![AK](https://img.shields.io/badge/media-AK-green)](https://x.com/_akhaliq/status/1765929288044290253?s=46)
[![MrNeRF](https://img.shields.io/badge/media-MrNeRF-green)](https://x.com/janusch_patas/status/1766446189749150126?s=46)
[![Hugging Face datasets](fig/badges/huggingface_datasets.svg)](https://huggingface.co/datasets/CaiYuanhao/X-Gaussian)
[![Hugging Face models](fig/badges/huggingface_models.svg)](https://huggingface.co/CaiYuanhao/X-Gaussian)

<h2> Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis </h2> 


<img src="3d_demo/teapot.gif" style="height:200px" /> 

<img src="3d_demo/foot.gif" style="height:160px" /> 

<img src="3d_demo/bonsai.gif" style="height:200px" /> 

Point Cloud Visualization

&nbsp;

<img src="3d_demo/training_process.gif" style="height:200px" /> 

Training Process Visualization

</div>


&nbsp;


### Introduction
This is the official implementation of our ECCV 2024 paper "Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis". Our X-Gaussian is SfM-free. If you find this repo useful, please give it a star ⭐ and consider citing our paper. Thank you.


### News
- **2026.11.09 :** We added real-data reconstruction on Pine, Seashell, and Walnut, with [data](https://huggingface.co/datasets/CaiYuanhao/X-Gaussian/tree/main/real) and [pretrained models](https://huggingface.co/CaiYuanhao/X-Gaussian) available on Hugging Face.
- **2026.11.09 :** We released the [datasets](https://huggingface.co/datasets/CaiYuanhao/X-Gaussian) and [pretrained models](https://huggingface.co/CaiYuanhao/X-Gaussian) on Hugging Face. 🤗
- **2026.11.08 :** We added direct 3D reconstruction.
- **2025.06.25 :** Our new work [X2-Gaussian](https://arxiv.org/abs/2503.21779) for dynamic human chest breathing CT reconstruction has been accepted by ICCV 2025. Congrats to [Weihao](https://yuyouxixi.github.io/). Code and models will be released at [this repo](https://github.com/yuyouxixi/x2-gaussian).  🚀
- **2024.09.01 :** Code, models, and training logs have been released. Welcome to have a try 😆
- **2024.07.01 :** Our X-Gaussian has been accepted by ECCV 2024! Code will be released before the start date of the conference (2024.09.29). Stay tuned. 🚀
- **2024.06.03 :** Code for traditional methods has been released at [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF). ✨
- **2024.06.03 :** Code for fancy visualization and data generation has been released at [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF). 🚀
- **2024.06.03 :** Data, code, models, and training logs of our CVPR 2024 work [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF) have been released. Feel free to use them :)
- **2024.06.03 :** The datasets have been released on [Google Drive](https://drive.google.com/drive/folders/1tBGulq6l-Gp_SI1UyylJgr9Zw01Oie3R?usp=sharing). Feel free to use them. 🚀
- **2024.03.07 :** Our paper is on [arxiv](https://arxiv.org/abs/2403.04116) now. Code, models, and training logs will be released. Stay tuned. 💫

### Performance

<details close>
<summary><b>Novel View Synthesis</b></summary>

![results1](/fig/nvs_1.png)

![results2](/fig/nvs_2.png)

</details>


<details close>
<summary><b>Direct 3D CT Reconstruction</b></summary>

Results on the original five datasets without downsampling, using 50 training projections,
30,000 training steps and 1,000 refinement steps. CT ground truth is used only for evaluation.

| Dataset | CT resolution | 2D PSNR (dB) | 2D SSIM | 3D PSNR (dB) | 3D SSIM |
|:--|:--|--:|--:|--:|--:|
| Chest | 128 × 128 × 128 | 47.21 | 0.9937 | 31.94 | 0.8895 |
| Abdomen | 512 × 512 × 463 | 48.16 | 0.9968 | 33.33 | 0.8917 |
| Foot | 256 × 256 × 256 | 44.40 | 0.9869 | 31.88 | 0.8795 |
| Head | 256 × 256 × 128 | 51.66 | 0.9979 | 40.21 | 0.9732 |
| Pancreas | 512 × 512 × 240 | 47.65 | 0.9923 | 22.22 | 0.8750 |

2D metrics average 50 held-out views, with each prediction and reference normalized by its own maximum.
3D metrics use a fixed range of 1 without clipping or intensity correction; SSIM averages nonempty reference slices over all three axes.

</details>


<details close>
<summary><b>Real Data 3D CT Reconstruction</b></summary>

Results on three real FIPS scans with 50 training and 100 held-out projections,
560 × 560 detector images, and a 256³ reconstruction grid. Each run uses
30,000 training steps and 1,000 refinement steps, without further downsampling.

| Scene | 2D PSNR (dB) | 2D SSIM | 3D PSNR vs. FDK (dB) | 3D SSIM vs. FDK |
|:--|--:|--:|--:|--:|
| Pine | 39.18 | 0.9777 | 37.79 | 0.9399 |
| Seashell | 32.54 | 0.9749 | 42.62 | 0.9593 |
| Walnut | 34.24 | 0.9140 | 30.32 | 0.6810 |

2D metrics average the 100 held-out measured projections, with each image
normalized by its own maximum. The 3D reference is **full-view FDK**, not measured
CT ground truth; 3D metrics use a fixed range of 1 without intensity correction.
The pseudo-reference and test views are not used for training or initialization.

![Real X-ray rendering and direct CT reconstruction](fig/real_data.png)

Left to right: measured X-ray, rendered X-ray, full-view FDK pseudo-reference,
and X-Gaussian CT. CT previews use the same fixed 0–1 intensity range.

</details>


<details close>
<summary><b>Assistant CT Reconstruction</b></summary>

![results3](/fig/ct_1.png)

![results4](/fig/ct_2.png)

</details>

### Coordinate System

The coordinate system in circular cone-beam X-ray scanning follows the OpenCV standards. The transformation between the camera, world, and image coordinate systems is shown below.
<div align="center">
<p align="center"> <img src="fig/coordinate_system.png" width="800px"> </p>
</div>

&nbsp;

## 1. Environment

Use Linux with a CUDA 11.8 toolkit (`nvcc`). The environment installs Python 3.9,
PyTorch 2.1.2 and GCC 11 for both 2D projection rendering and 3D CT reconstruction.

```shell
git clone --recursive https://github.com/caiyuanhao1998/X-Gaussian.git
cd X-Gaussian
conda env create -f environment.yml
conda activate x_gaussian
bash install.sh
```

## 2. Dataset

Download the original five CT datasets and three real scans from [Hugging Face](https://huggingface.co/datasets/CaiYuanhao/X-Gaussian).
These datasets are also available on [Google Drive](https://drive.google.com/drive/folders/1tBGulq6l-Gp_SI1UyylJgr9Zw01Oie3R?usp=sharing).
With [Git LFS](https://git-lfs.com/) installed, run from the project root:

```shell
git lfs install
git clone https://huggingface.co/datasets/CaiYuanhao/X-Gaussian data
```

The processed data files are placed in `data/`:

```text
|--data
      |--synthetic
            # The first five datasets are used in our paper
            |--chest_50.pickle
            |--abdomen_50.pickle
            |--foot_50.pickle
            |--head_50.pickle
            |--pancreas_50.pickle
      |--real
            |--pine
            |--seashell
            |--walnut
```

Training reads these files directly and initializes from training projections.
Datasets and outputs can be stored elsewhere and symlinked into `data/` and `output/`.

Each real scene contains `meta_data.json`, 50 training and 100 test projections at
560 × 560, a 256³ full-view FDK pseudo-reference, and training-only initialization.
We preserve the released processed resolution without further downsampling.
See the [real-data sources and format](https://huggingface.co/datasets/CaiYuanhao/X-Gaussian/blob/main/real/README.md).

## 3. Testing

### Prepare Pretrained Models
Please download the pretrained models from [Hugging Face](https://huggingface.co/CaiYuanhao/X-Gaussian) using Git LFS:

```shell
git lfs install
git clone https://huggingface.co/CaiYuanhao/X-Gaussian pretrained
```

Our models can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/1TmpuxMgKufvn_ltq0Z16smQfyKZrhR2F?usp=sharing). Please organize the models in `pretrained/` as
```text
|--pretrained/
      |--synthetic
            # The first five datasets are used in our paper
            |--chest
            |--abdomen
            |--foot
            |--head
            |--pancreas
      |--real
            |--pine
            |--seashell
            |--walnut
```

### 2D Projection Rendering

```shell
# synthetic scenes
python render.py -m pretrained/synthetic/chest --mode 2d
python render.py -m pretrained/synthetic/abdomen --mode 2d
python render.py -m pretrained/synthetic/foot --mode 2d
python render.py -m pretrained/synthetic/head --mode 2d
python render.py -m pretrained/synthetic/pancreas --mode 2d

# real scenes
python render.py -m pretrained/real/pine --mode 2d
python render.py -m pretrained/real/seashell --mode 2d
python render.py -m pretrained/real/walnut --mode 2d
```

Renders test cameras to each model folder’s `projections_test/` (raw NPY and PNG previews).

### 3D Reconstruction Testing

```shell
# synthetic scenes
python render.py -m pretrained/synthetic/chest --mode 3d
python render.py -m pretrained/synthetic/abdomen --mode 3d
python render.py -m pretrained/synthetic/foot --mode 3d
python render.py -m pretrained/synthetic/head --mode 3d
python render.py -m pretrained/synthetic/pancreas --mode 3d

# real scenes
python render.py -m pretrained/real/pine --mode 3d
python render.py -m pretrained/real/seashell --mode 3d
python render.py -m pretrained/real/walnut --mode 3d
```

Saves `volume.npy` in each model folder. Use `--output` to select another path.

The same trained Gaussians directly define the CT volume:

$$
V(\mathbf x)=\sum_i\rho_i\exp\!\left[-\frac12
(\mathbf x-\mathbf p_i)^\top\Sigma_i^{-1}(\mathbf x-\mathbf p_i)\right].
$$

Here $\rho_i=\sigma(o_i)\bar r_i$ combines the original opacity and positive radiative
intensity, and $\Sigma_i=R(q_i)D_iR(q_i)^\top$
uses the original rotations and squared scales in the diagonal matrix $D_i$. Reconstruction samples this field at
voxel centers; no extra learned density or voxel parameters are introduced.

3D mode also accepts a standalone CT-trained PLY via `-m path/to/point_cloud.ply`,
without projections or CT truth. The saved scanner resolution and extent are used by default; `--resolution`,
`--center` and `--size` override them. Both modes load the latest
checkpoint by default (`--iteration` selects another checkpoint).

## 4. Training

```shell
# synthetic scenes
python train.py --config config/synthetic/chest.yaml
python train.py --config config/synthetic/abdomen.yaml
python train.py --config config/synthetic/foot.yaml
python train.py --config config/synthetic/head.yaml
python train.py --config config/synthetic/pancreas.yaml

# real scenes
python train.py --config config/real/pine.yaml
python train.py --config config/real/seashell.yaml
python train.py --config config/real/walnut.yaml
```

Checkpoints are saved to `output/synthetic/<scene>/` or `output/real/<scene>/`.
Training reuses cached initialization when available; otherwise it computes and
saves FDK initialization automatically from the training projections.

Training jointly optimizes one Gaussian field for 2D projection rendering and 3D
reconstruction, using projection loss, 3D TV, and volume-ray consistency. The default
recipe runs 30,000 base steps and 1,000 refinement steps. CT ground truth is not used
for training. The original Gaussian attributes are retained.

## Citation
```sh
# X-Gaussian
@inproceedings{x_gaussian,
  title={Radiative gaussian splatting for efficient x-ray novel view synthesis},
  author={Yuanhao Cai and Yixun Liang and Jiahao Wang and Angtian Wang and Yulun Zhang and Xiaokang Yang and Zongwei Zhou and Alan Yuille},
  booktitle={ECCV},
  year={2024}
}

# sax-nerf
@inproceedings{sax_nerf,
  title={Structure-Aware Sparse-View X-ray 3D Reconstruction},
  author={Yuanhao Cai and Jiahao Wang and Alan Yuille and Zongwei Zhou and Angtian Wang},
  booktitle={CVPR},
  year={2024}
}

# X2-Gaussian
@inproceedings{x2_gaussian,
  title={X2-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction},
  author={Yu, Weihao and Cai, Yuanhao and Zha, Ruyi and Fan, Zhiwen and Li, Chenxin and Yuan, Yixuan},
  booktitle={ICCV},
  year={2025}
}
```

&nbsp;


## Acknowledgement

Our code and data are heavily borrowed from [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF/) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).
