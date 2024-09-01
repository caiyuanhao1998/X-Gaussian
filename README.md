&nbsp;

<div align="center">

<p align="center"> <img src="3d_demo/logo.png" width="110px"> </p>

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2403.04116)
[![Youtube](https://img.shields.io/badge/video-youtube-red)](https://www.youtube.com/watch?v=gDVf_Ngeghg)

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
