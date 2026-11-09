"""Data-free cone geometry for analytic CUDA and ray-integration checks."""
from types import SimpleNamespace
import numpy as np
import torch
from xct.data import angle2pose
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def phantom_data():
    scanner = dict(DSO=5.0, DSD=7.0, sDetector=np.array([4., 4.]),
                   sVoxel=np.array([2., 2., 2.]), nVoxel=np.array([256]*3),
                   offOrigin=np.zeros(3), dVoxel=np.array([2/256]*3))
    fov = 2*np.arctan2(2., scanner['DSD'])
    image = torch.zeros(1, 512, 512, device='cuda')
    projection = getProjectionMatrix(.01, 100., fov, fov).T.cuda()
    cameras = []
    for i in range(50):
        w2c = np.linalg.inv(angle2pose(scanner['DSO'], i*2*np.pi/50))
        view = torch.tensor(getWorld2View2(w2c[:3,:3].T,w2c[:3,3])).T.cuda()
        cameras.append(SimpleNamespace(image_name=f'phantom_{i}', uid=i,
            original_image=image, image_height=512, image_width=512, mode=1,
            FoVx=fov, FoVy=fov, world_view_transform=view,
            full_proj_transform=view@projection,
            camera_center=torch.linalg.inv(view)[3,:3]))
    return SimpleNamespace(train=cameras, scanner=scanner)
