&nbsp;

<div align="center">

<p align="center"> <img src="3d_demo/logo.png" width="110px"> </p>

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2403.04116)
[![zhihu](https://img.shields.io/badge/知乎-解读-179bd3)](https://zhuanlan.zhihu.com/p/717744222)
[![Youtube](https://img.shields.io/badge/video-youtube-red)](https://www.youtube.com/watch?v=v6FESb3SkJg&t=28s)
[![AK](https://img.shields.io/badge/media-AK-green)](https://x.com/_akhaliq/status/1765929288044290253?s=46)
[![MrNeRF](https://img.shields.io/badge/media-MrNeRF-green)](https://x.com/janusch_patas/status/1766446189749150126?s=46)
[![RF](https://img.shields.io/badge/media-Radiance_Fields-green)](https://radiancefields.com/x-gaussian-radiance-meets-radiation)

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
- **2025.06.25 :** Our new work [X2-Gaussian](https://arxiv.org/abs/2503.21779) for dynamic human chest breathing CT reconstruction has been accepted by ICCV 2025. Congrats to [Weihao](https://yuyouxixi.github.io/). Code and models will be released at [this repo](https://github.com/yuyouxixi/x2-gaussian).  🚀
- **2024.09.25 :** Our new work [R2-Gaussian](https://arxiv.org/abs/2405.20693v1) has been accepted by NeurIPS 2024. Congrats to [Ruyi](https://ruyi-zha.github.io/). Code and model are  released at [this repo](https://github.com/Ruyi-Zha/r2_gaussian). 💫 
- **2024.09.01 :** Code, models, and training logs have been released. Welcome to have a try 😆
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
cd X-Gaussian
conda env create --file environment.yml
conda activate x_gaussian

# Use our X-ray rasterizer package to replace the original RGB rasterizer
rm -rf submodules/diff-gaussian-rasterization/cuda_rasterizer
mv cuda_rasterizer submodules/diff-gaussian-rasterization/

# re-install the diff-gaussian-rasterization package
pip install submodules/diff-gaussian-rasterization
```

&nbsp;


## 2. Prepare Dataset
Download our processed datasets from [Google drive](https://drive.google.com/drive/folders/1W46wpeN7byWLC0f3cGIvoT_xbwT1b7gZ?usp=sharing) or [Baidu disk](https://pan.baidu.com/s/1WrYhxFb8Y-RwS4PCx_LRhA?pwd=cyh2). Then put the downloaded datasets into the folder `data/` as

```sh
  |--data
      # The first five datasets are used in our paper
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

`Note:` The first five datasets are used to do experiments in our paper. The rest datasets are from [the X3D benchmark](https://github.com/caiyuanhao1998/SAX-NeRF/). Please also note that the pickle data used by X-Gaussian is dumped/read by pickle protocol 4, which is supported by python < 3.8. The original X3D data is processed by pickle protocol 5, which is supported by python >= 3.8. I have re-dumped the pickle data from the original X3D datasets to make sure you can run our code without extra effort. If you want to re-dump the pickle data, please run

```shell
python pickle_redump.py
```


&nbsp;

## 3. Training and Testing

You can download our trained Gaussian point clouds from [Google Drive](https://drive.google.com/drive/folders/1-JqRXiwl1zjVKuBRL3F01cWHcyAe8f2F?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1GWE5By6u03n2l6nnFhOE0g?pwd=cyh2) (code: `cyh2`) as

![pc_shape](/fig/point_cloud_shape.png)

We share the training log for your convienience to debug. Please download them from [Google Drive](https://drive.google.com/drive/folders/1HKcy-luYLXTSH7vviu_djVtpbSdz_v6J?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1amk0tnpH3hN9I-Qgjb_cEQ?pwd=cyh2) (code: `cyh2`). To make the training and evaluation easier, your can directly run the `train.sh` file by

```shell
bash train.sh
```

Or you can separately train on each scene like

```shell

python3 train.py --config config/chest.yaml --eval

python3 train.py --config config/foot.yaml --eval

python3 train.py --config config/abdomen.yaml --eval

python3 train.py --config config/head.yaml --eval

python3 train.py --config config/pancreas.yaml --eval

python3 train.py --config config/jaw.yaml --eval

python3 train.py --config config/pelvis.yaml --eval

python3 train.py --config config/aneurism.yaml --eval

python3 train.py --config config/carp.yaml --eval

python3 train.py --config config/bonsai.yaml --eval

python3 train.py --config config/box.yaml --eval

python3 train.py --config config/backpack.yaml --eval

python3 train.py --config config/engine.yaml --eval

python3 train.py --config config/leg.yaml --eval

python3 train.py --config config/teapot.yaml --eval

```

&nbsp;


## 4. Visualization

We also provide code for the visualization of rotating the Gaussian point clouds

```shell

python point_cloud_vis.py

```

&nbsp;

## 5. Citation
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

# R2-Gaussian
@inproceedings{r2_gaussian,
  title={R2-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction},
  author={Ruyi Zha and Tao Jun Lin and Yuanhao Cai and Jiwen Cao and Yanhao Zhang and Hongdong Li},
  booktitle={NeurIPS},
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

Our code and data are heavily borrowed from [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF/) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
