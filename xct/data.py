"""Original-resolution NeRF-format CT data. Geometry follows R2 conventions.

The GT volume is read only by load_ground_truth(), called during evaluation.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def angle2pose(dso, angle):
    # Rz(angle) Rz(pi/2) Rx(-pi/2), identical to the official scanner convention.
    c, s = np.cos(angle), np.sin(angle)
    rx = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    r = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pose = np.eye(4)
    pose[:3,:3] = r @ rz @ rx
    pose[:3,3] = [dso*c, dso*s, 0]
    return pose


class CTData:
    def __init__(self, path, device='cuda', load_test=True, load_images=True):
        self.path = Path(path).resolve()
        self.device = device
        self.load_images = load_images
        self.meta = json.loads((self.path/'meta_data.json').read_text())
        self.scanner = dict(self.meta['scanner'])
        self.scene_scale = 2 / max(self.scanner['sVoxel'])
        for k in ['DSD','DSO','sVoxel','sDetector','offOrigin','offDetector']:
            self.scanner[k] = np.asarray(self.scanner[k]) * self.scene_scale
        self.scanner['dVoxel'] = self.scanner['sVoxel']/self.scanner['nVoxel']
        self.bbox = torch.tensor(np.stack([
            self.scanner['offOrigin']-self.scanner['sVoxel']/2,
            self.scanner['offOrigin']+self.scanner['sVoxel']/2]), dtype=torch.float32)
        self.train = self._cameras('train')
        self.test = self._cameras('test') if load_test else []

    def _cameras(self, split):
        cfg = self.scanner
        if cfg['mode'] != 'cone':
            raise ValueError('This CT data adapter currently supports cone-beam metadata')
        fx = 2*np.arctan2(cfg['sDetector'][1]/2,cfg['DSD'])
        fy = 2*np.arctan2(cfg['sDetector'][0]/2,cfg['DSD'])
        cameras = []
        for i, frame in enumerate(self.meta['proj_'+split]):
            pose = angle2pose(cfg['DSO'],frame['angle'])
            w2c = np.linalg.inv(pose)
            view = torch.tensor(getWorld2View2(w2c[:3,:3].T,w2c[:3,3])).T.to(self.device)
            projection = getProjectionMatrix(.01,100.,fx,fy).T.to(self.device)
            image = (np.load(self.path/frame['file_path']).astype(np.float32)*self.scene_scale
                     if self.load_images else np.zeros(cfg['nDetector'],dtype=np.float32))
            if list(image.shape) != list(cfg['nDetector']):
                raise ValueError('Projection/metadata resolution mismatch')
            cameras.append(SimpleNamespace(
                image_name=Path(frame['file_path']).stem, uid=i,
                original_image=torch.from_numpy(image)[None].to(self.device),
                image_height=image.shape[0], image_width=image.shape[1],
                mode=1, FoVx=fx, FoVy=fy, world_view_transform=view,
                full_proj_transform=view@projection,
                camera_center=torch.linalg.inv(view)[3,:3]))
        return cameras

    def initialization(self):
        points = np.load(self.path/('init_'+self.path.name+'.npy'))
        # Official init is already expressed in the normalized scanner coordinate system.
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError('Invalid official initialization')
        return points
