"""Original-resolution cone-beam CT data; training never uses the reference volume."""
import json
import pickle
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
    def __init__(self, path, device='cuda', load_test=True, load_images=True, cache_dir=None):
        self.path = Path(path).resolve()
        self.device = device
        self.load_images = load_images
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.projections = None
        if self.path.is_file():
            self.meta, self.projections = self._read_pickle(load_test)
        else:
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

    def _read_pickle(self, load_test):
        with self.path.open('rb') as stream:
            raw = pickle.load(stream)
        raw.pop('image', None)  # The reference CT is not used for training/initialization.
        # Original files store scanner lengths in mm and projection integrals in m.
        scanner = {k: raw[k] for k in ['nDetector', 'nVoxel', 'mode']}
        for key in ['DSD', 'DSO', 'offOrigin', 'offDetector']:
            scanner[key] = (np.asarray(raw[key]) / 1000).tolist()
        scanner['sVoxel'] = (np.asarray(raw['nVoxel']) * np.asarray(raw['dVoxel']) / 1000).tolist()
        scanner['sDetector'] = (np.asarray(raw['nDetector']) * np.asarray(raw['dDetector']) / 1000).tolist()
        meta = dict(scanner=scanner)
        projections = {}
        for split in ['train'] + (['test'] if load_test else []):
            part = raw['val' if split == 'test' and 'val' in raw else split]
            meta['proj_' + split] = [dict(angle=float(angle), file_path=f'{split}_{i:04d}.npy')
                                     for i, angle in enumerate(part['angles'])]
            if self.load_images:
                projections[split] = part['projections']
        return meta, projections

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
            if self.load_images:
                raw = self.projections[split][i] if self.projections is not None else np.load(self.path/frame['file_path'])
                image = np.asarray(raw, dtype=np.float32) * self.scene_scale
            else:
                image = np.zeros(cfg['nDetector'], dtype=np.float32)
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
        if self.path.is_dir():
            path = self.path / ('init_' + self.path.name + '.npy')
        else:
            path = self.path.with_name('init_' + self.path.stem + '.npy')
        if not path.exists():
            if self.cache_dir is None:
                raise ValueError('Set cache_dir to save the training-only FDK initialization')
            path = self.cache_dir / 'initialization.npy'
            if not path.exists():
                from xct.initialization import fdk_points
                print('Initializing from training projections at original CT resolution...', flush=True)
                points = fdk_points(self.train, self.scanner, volume_path=self.cache_dir/'fdk_volume.npy')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, points)
        points = np.load(path)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError('Invalid initialization')
        return points
