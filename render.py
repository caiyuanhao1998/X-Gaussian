"""Render 2D X-ray projections or directly reconstruct a 3D CT volume."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from scene.gaussian_model import GaussianModel_Xray
from xct.data import CTData
from xct.field import query, render
from xct.io import configure_from_ply, field_metadata


def load_model(path, iteration=-1):
    path = Path(path)
    if path.is_file():
        model = path
        config_path = path.parents[2] / 'config.json' if len(path.parents) > 2 else None
    else:
        if iteration < 0:
            checkpoints = list((path / 'point_cloud').glob('iteration_*/point_cloud.ply'))
            if not checkpoints:
                raise FileNotFoundError(f'No Gaussian checkpoint in {path}')
            model = max(checkpoints, key=lambda p: int(p.parent.name.split('_')[-1]))
        else:
            model = path / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
        config_path = path / 'config.json'
    config = json.loads(config_path.read_text()) if config_path and config_path.exists() else {}
    meta = field_metadata(model)
    pc = GaussianModel_Xray(meta.get('sh_degree', config.get('sh_degree', 3)))
    pc.load_ply(str(model))
    configure_from_ply(pc, model, config)
    return pc, config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_path', required=True, help='Model directory or standalone CT-trained PLY')
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d')
    parser.add_argument('--iteration', type=int, default=-1, help='Defaults to the latest checkpoint')
    parser.add_argument('-s', '--source_path', help='Camera dataset for 2D rendering; defaults to the saved config')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--output', type=Path, help='2D output directory or 3D .npy filename')
    parser.add_argument('--resolution', type=int, nargs=3, default=None)
    parser.add_argument('--size', type=float, nargs=3, default=None)
    parser.add_argument('--center', type=float, nargs=3, default=None)
    args = parser.parse_args()
    pc, config = load_model(args.model_path, args.iteration)
    geometry = getattr(pc, 'volume_geometry', config.get('volume_geometry', {}))
    args.resolution = args.resolution or geometry.get('nVoxel', [256, 256, 256])
    args.size = args.size or geometry.get('sVoxel', [2., 2., 2.])
    args.center = args.center or geometry.get('offOrigin', [0., 0., 0.])
    if min(args.resolution) <= 0 or min(args.size) <= 0:
        parser.error('Resolution and size must be positive')
    model_root = Path(args.model_path)
    if model_root.is_file():
        model_root = model_root.parent
    if args.mode == '3d':
        output = args.output or model_root / 'volume.npy'
        if output.suffix != '.npy':
            parser.error('The 3D output must be a .npy filename')
        volume = query(pc, args.center, args.resolution, args.size)['vol']
        if not torch.isfinite(volume).all():
            raise FloatingPointError('Nonfinite CT volume')
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, volume.cpu().numpy())
        output.with_suffix('.json').write_text(json.dumps(dict(axis_order='xyz', sample_location='voxel centers',
            resolution=args.resolution, size=args.size, center=args.center,
            spacing=(np.array(args.size) / args.resolution).tolist()), indent=2) + '\n')
        print(f'Saved 3D CT volume {tuple(volume.shape)} to {output}')
    else:
        source = args.source_path or config.get('source_path')
        if not source:
            parser.error('2D rendering requires --source_path or a saved model config')
        # Camera metadata is sufficient: rendering does not read target projections or CT truth.
        data = CTData(source, load_test=args.split == 'test', load_images=False)
        cameras = data.test if args.split == 'test' else data.train
        output = args.output or model_root / f'projections_{args.split}'
        output.mkdir(parents=True, exist_ok=True)
        for camera in tqdm(cameras, desc='Rendering 2D projections'):
            image = render(camera, pc)['render'].squeeze(0).cpu().numpy()
            if not np.isfinite(image).all():
                raise FloatingPointError('Nonfinite X-ray projection')
            np.save(output / f'{camera.image_name}.npy', image)
            # PNG is a display preview; NPY retains the unnormalized attenuation integral.
            preview = np.clip(image / max(float(image.max()), 1e-12) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(preview).save(output / f'{camera.image_name}.png')
        print(f'Saved {len(cameras)} 2D projections to {output}')


if __name__ == '__main__':
    main()
