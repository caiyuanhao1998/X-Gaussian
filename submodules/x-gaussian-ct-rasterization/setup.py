#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

glm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
    '..', 'diff-gaussian-rasterization', 'third_party', 'glm'))
if not os.path.isfile(os.path.join(glm_dir, 'glm', 'glm.hpp')):
    raise RuntimeError('Initialize the rasterizer submodule first: git submodule update --init --recursive')

setup(
    name="x_gaussian_ct_rasterization",
    packages=["x_gaussian_ct_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="x_gaussian_ct_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "cuda_voxelizer/voxelizer_impl.cu",
                "cuda_voxelizer/forward.cu",
                "cuda_voxelizer/backward.cu",
                "voxelize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": ["-I" + glm_dir]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
