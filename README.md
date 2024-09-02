&nbsp;

<div align="center">

<p align="center"> <img src="3d_demo/logo.png" width="110px"> </p>

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2403.04116)
[![Youtube](https://img.shields.io/badge/video-youtube-red)](https://www.youtube.com/watch?v=v6FESb3SkJg&t=28s)

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
- **2024.09.01 :** Code have been released. Welcome to have a try 😆
- **2024.07.01 :** Our X-Gaussian has been accepted by ECCV 2024! Code will be released before the start date of the conference (2024.09.29). Stay tuned. 🚀
- **2024.06.03 :** Code for traditional methods has been released at [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF). ✨
- **2024.06.03 :** Code for fancy visualization and data generation has been released at [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF). 🚀
- **2024.06.03 :** Data, code, models, and training logs of our CVPR 2024 work [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF) have been released. Feel free to use them :)
- **2024.06.03 :** The datasets have been released on [Google Drive](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing). Feel free to use them. 🚀
- **2024.03.07 :** Our paper is on [arxiv](https://arxiv.org/abs/2403.04116) now. Code, models, and training logs will be released. Stay tuned. 💫

### Performance

<details close>
<summary><b>Novel View Synthesis</b></summary>

![results1](/fig/nvs_1.png)

![results2](/fig/nvs_2.png)

</details>


<details close>
<summary><b>CT Reconstruction</b></summary>

![results3](/fig/ct_1.png)

![results4](/fig/ct_2.png)

</details>

### Coordinate System

The coordinate system in circular cone-beam X-ray scanning follows the OpenCV standards. The transformation between the camera, world, and image coordinate systems is shown below.
<div align="center">
<p align="center"> <img src="fig/coordinate_system.png" width="800px"> </p>
</div>

&nbsp;

## 1. Create Environment:

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up the environment.

``` sh
# cloning our repo
git clone https://github.com/caiyuanhao1998/X-Gaussian --recursive


SET DISTUTILS_USE_SDK=1 # Windows only

# install the official environment of 3DGS
conda env create --file environment.yml
conda activate x_gaussian

# Then put our rasterizer package into the diff-gaussian-rasterization
mv cuda_rasterizer submodules/diff-gaussian-rasterization/

# re-install the diff-gaussian-rasterization package
cd submodules
pip install diff-gaussian-rasterization

# back to the main folder
cd ..
```

&nbsp;


## Prepare Dataset
Download our processed datasets from [Google drive](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing) or [Baidu disk](https://pan.baidu.com/s/18zc6jHeNvoUNAF6pUaL9eQ?pwd=cyh2). Then put the downloaded datasets into the folder `data/` as

```sh
  |--data
      # The first five datasets are used in the our paper
      |--chest_50.pickle
      |--abdomen_50.pickle
      |--foot_50.pickle
      |--head_50.pickle
      |--pancreas_50.pickle
      # The rest datasets are from the X3D benchmark
      |--aneurism_50.pickle
      |--backpack_50.pickle
      |--bonsai_50.pickle
      |--box_50.pickle
      |--carp_50.pickle
      |--engine_50.pickle
      |--leg_50.pickle
      |--pelvis_50.pickle
      |--teapot_50.pickle
      |--jaw_50.pickle
```

`Note:` The first five datasets are used to do experiments in our paper. The rest datasets are from [the X3D benchmark](https://github.com/caiyuanhao1998/SAX-NeRF/).

&nbsp;

## Training and Testing

```shell
bash train.sh
```

&nbsp;

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
```

&nbsp;


## Acknowledgement

Our code and data are heavily borrowed from [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF/) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
