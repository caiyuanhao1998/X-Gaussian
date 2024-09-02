from plyfile import PlyData, PlyElement
import numpy as np
from pdb import set_trace as stx
import matplotlib.pyplot as plt
import os
from utils.image_utils import min_max_norm
from tqdm import tqdm
from PIL import Image

scene = 'foot'
method = 'XGaussian'

save_path = f'/home/ycai51/XGaussian_sinc_nodirec_norm/point_cloud_visualization/{scene}_no_norm/'
# save_path = f'point_cloud_visualization/{scene}/'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

os.makedirs(save_path, exist_ok=True)

# path = 'output/foot/2024_01_10_23_04_44/point_cloud/iteration_30000/point_cloud.ply'
path = 'output/foot/2024_01_31_10_00_42/point_cloud/iteration_30000/point_cloud.ply'

plydata = PlyData.read(path)

xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
points = xyz
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=min_max_norm(opacities), color='gray')

# 取消三维网格坐标系
ax.axis('off')

# 上下翻转三维图像
ax.invert_zaxis()

elevation = 0
azimuth = 0

proj_num = 120
angle_interval = 360 / proj_num

image_files = []

for i in tqdm(range(proj_num)):
    angle = i * angle_interval
    ax.view_init(elev=elevation, azim=angle)
    plt.savefig(save_path + f'elev_{elevation}_azim_{angle}.png', dpi=500, bbox_inches='tight')
    image_files.append(save_path + f'elev_{elevation}_azim_{angle}.png')


# stx()
box = (300, 500, 1700, 1350)

fps = 30
duration = 1000 / fps
gif_filename = os.path.join(save_path, f'{scene}_fps_{fps}.gif')


img = Image.open(image_files[0])

gif_frames = [img.crop(box)]

# 逐一添加图像帧
for filename in tqdm(image_files[1:]):
    img = Image.open(filename)
    gif_frames.append(img.crop(box))

gif_frames[0].save(gif_filename, save_all=True, append_images=gif_frames[1:], duration=duration, loop=0)