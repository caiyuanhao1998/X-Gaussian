"""Training-only FDK, using SAX-NeRF scanner units and image-axis conventions."""
import math
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def fdk_volume(cameras, scanner, chunk_size=524288):
    device = cameras[0].original_image.device
    h, w = cameras[0].image_height, cameras[0].image_width
    dsd, dso = float(scanner['DSD']), float(scanner['DSO'])
    sv, su = map(float, scanner['sDetector'])
    u = (torch.arange(w, device=device) + .5 - w / 2) * su / w
    v = (torch.arange(h, device=device) + .5 - h / 2) * sv / h
    cosine = dsd / torch.sqrt(dsd * dsd + v[:, None] ** 2 + u[None, :] ** 2)
    padded = 1 << math.ceil(math.log2(2 * w))
    # Discrete Ram-Lak, as in SAX-NeRF/TIGRE (retains the finite-filter DC term).
    coordinate = torch.arange(-padded // 2, padded // 2, device=device)
    kernel = torch.zeros(padded, device=device)
    kernel[padded // 2] = .25
    odd = coordinate % 2 != 0
    kernel[odd] = -1 / (math.pi * coordinate[odd].float()) ** 2
    ramp = torch.fft.rfft(torch.fft.ifftshift(kernel)).abs() / (su / w)
    filtered = []
    for camera in cameras:
        projection = camera.original_image[0] * cosine
        spectrum = torch.fft.rfft(projection, n=padded, dim=-1)
        filtered.append(torch.fft.irfft(spectrum * ramp, n=padded, dim=-1)[:, :w])
    nx, ny, nz = map(int, scanner['nVoxel'])
    size = torch.as_tensor(scanner['sVoxel'], device=device, dtype=torch.float32)
    spacing = size / torch.tensor([nx, ny, nz], device=device)
    low = torch.as_tensor(scanner['offOrigin'], device=device, dtype=torch.float32) - size / 2
    volume = torch.empty(nx * ny * nz, device=device)
    for start in range(0, len(volume), chunk_size):
        index = torch.arange(start, min(start + chunk_size, len(volume)), device=device)
        ijk = torch.stack([index // (ny * nz), (index // nz) % ny, index % nz], 1)
        xyz = low + (ijk.float() + .5) * spacing
        values = torch.zeros(len(index), device=device)
        for camera, projection in zip(cameras, filtered):
            view = camera.world_view_transform
            local = xyz @ view[:3, :3] + view[3, :3]
            z = local[:, 2].clamp_min(1e-6)
            grid = torch.stack([local[:, 0] / z / math.tan(camera.FoVx / 2),
                                local[:, 1] / z / math.tan(camera.FoVy / 2)], -1)
            samples = F.grid_sample(projection[None, None], grid[None, None],
                                    align_corners=False, padding_mode='zeros').reshape(-1)
            values += samples * (dso / z) ** 2
        volume[start:start + len(index)] = values * (math.pi / len(cameras)) * (dsd / dso)
    return volume.reshape(nx, ny, nz)


def fdk_points(cameras, scanner, count=50000, volume_path=None):
    volume = fdk_volume(cameras, scanner).cpu().numpy()
    if not np.isfinite(volume).all():
        raise FloatingPointError('Nonfinite training-only FDK initialization')
    if volume_path is not None:
        from pathlib import Path
        Path(volume_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(volume_path, volume)
    valid = np.flatnonzero(volume > .05)
    if not len(valid):
        raise ValueError('No positive FDK support; check scanner geometry and projection units')
    selected = np.random.RandomState(0).choice(valid, min(count, len(valid)), replace=False)
    ijk = np.stack(np.unravel_index(selected, volume.shape), 1)
    size = np.asarray(scanner['sVoxel'])
    xyz = np.asarray(scanner['offOrigin']) - size / 2 + (ijk + .5) * size / np.asarray(scanner['nVoxel'])
    density = volume.reshape(-1)[selected, None] * .15
    return np.concatenate([xyz, density], 1).astype(np.float32)


def save_initialization(pc, data, directory, init_source=None):
    """Keep reusable points, initial Gaussians, and scanner metadata with each run."""
    import json
    from pathlib import Path
    from xct.io import save_ct_ply
    directory=Path(directory)
    save_ct_ply(pc,directory/'initial_gaussians.ply')
    metadata=dict(method='provided' if init_source else 'FDK',filter='ram_lak',source_path=str(data.path),
        training_views=len(data.train),ground_truth_used=False,downsampling=False,
        point_count=len(pc.get_xyz),sampling_seed=0,density_scale=.15,
        points='initialization.npy',point_columns=['x','y','z','density'],
        gaussians='initial_gaussians.ply',volume_geometry=pc.volume_geometry,
        fdk_volume='fdk_volume.npy' if (directory/'fdk_volume.npy').exists() else None,
        init_source=str(init_source) if init_source else None)
    (directory/'initialization.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    import argparse
    import random
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    from xct.data import CTData
    from xct.refinement import project_scales
    from train import initialize
    parser=argparse.ArgumentParser(description='Generate training-only FDK and Gaussian initialization')
    parser.add_argument('--config',default='config/synthetic/chest.yaml')
    parser.add_argument('--source_path')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if (args.output/'initialization.json').exists():
        parser.error('Initialization already exists; choose another --output or reuse initialization.npy')
    args.output.mkdir(parents=True,exist_ok=True)
    data=CTData(args.source_path or cfg['source_path'],load_test=False,cache_dir=args.output)
    pc=initialize(data,cfg['sh_degree'])
    pc.volume_geometry={k:np.asarray(data.scanner[k]).tolist() for k in ['nVoxel','sVoxel','offOrigin']}
    project_scales(pc,SimpleNamespace(**cfg))
    save_initialization(pc,data,args.output)
    print('Initialization saved to',args.output)


if __name__=='__main__':
    main()
