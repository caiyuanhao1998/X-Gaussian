"""Evaluate exported CT volumes and rendered projections at their stored resolution."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from utils.loss_utils import ssim
from xct.data import CTData


@torch.no_grad()
def volume_metrics(volume, reference):
    predicted = np.load(volume, mmap_mode='r')
    if predicted.shape != reference.shape:
        raise ValueError(f'Volume shapes differ: {predicted.shape} and {reference.shape}')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError('Volume contains nonfinite values')
    mse = sum(float(np.square(predicted[i].astype(np.float64)-reference[i]).sum())
              for i in range(len(predicted))) / predicted.size
    p = torch.from_numpy(np.array(predicted)).float().cuda()
    g = torch.from_numpy(np.array(reference)).float().cuda()
    axes = []
    for axis in range(3):
        a, b = p.movedim(axis, 0), g.movedim(axis, 0)
        scores = []
        for start in range(0, len(a), 8):
            v, w = a[start:start+8, None], b[start:start+8, None]
            mask = w.amax((1, 2, 3)) > 0
            if mask.any():
                scores.extend(ssim(v[mask], w[mask], size_average=False).cpu().tolist())
        axes.append(float(np.mean(scores)))
    return dict(psnr_3d=float(-10*np.log10(mse)), ssim_3d=float(np.mean(axes)),
                ssim_axes=axes, volume_shape=list(predicted.shape), data_range=1,
                clipping=False, intensity_correction=False)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-m', '--model_path', type=Path, required=True)
    p.add_argument('-s', '--source_path', type=Path)
    p.add_argument('--mode', choices=['2d', '3d', 'all'], default='all')
    p.add_argument('--projection_renderer', choices=['raster', 'volume'], default='raster')
    p.add_argument('--volume', type=Path)
    p.add_argument('--projections', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    config = json.loads((args.model_path/'config.json').read_text())
    source = args.source_path or Path(config['source_path'])
    result = dict(model=str(args.model_path), source_path=str(source),
                  experiment='CT extension; distinct from the original paper NVS experiment',
                  downsampling=False)
    if args.mode in ['3d', 'all']:
        if source.is_file():
            with source.open('rb') as f:
                reference = pickle.load(f)['image']
            result['reference_3d'] = 'Original CT volume'
        else:
            meta = json.loads((source/'meta_data.json').read_text())
            reference = np.load(source/meta['vol'], mmap_mode='r')
            result['reference_3d'] = 'Full-view FDK pseudo-reference'
        result.update(volume_metrics(args.volume or args.model_path/'volume.npy', reference))
        del reference
        torch.cuda.empty_cache()
    if args.mode in ['2d', 'all']:
        data = CTData(source, load_test=True)
        suffix = '_volume' if args.projection_renderer == 'volume' else ''
        directory = args.projections or args.model_path/('projections_test'+suffix)
        scores, similarities, raw = [], [], []
        for camera in data.test:
            image = torch.as_tensor(np.load(directory/(camera.image_name+'.npy')), device='cuda')[None]
            target = camera.original_image
            if image.shape != target.shape or not torch.isfinite(image).all():
                raise ValueError('Projection shape or values are invalid')
            raw.append(float(-10*torch.log10(((image-target)/data.scene_scale).square().mean())))
            image = image/image.amax().clamp_min(1e-12)
            target = target/target.amax().clamp_min(1e-12)
            scores.append(float(-10*torch.log10((image-target).square().mean())))
            similarities.append(float(ssim(image, target)))
        result.update(psnr_2d=float(np.mean(scores)), ssim_2d=float(np.mean(similarities)),
                      raw_psnr_2d=float(np.mean(raw)), test_views=len(scores),
                      projection_renderer=args.projection_renderer,
                      normalization='Each prediction/reference projection divided by its own maximum')
    output = args.output or args.model_path/('evaluation_'+args.projection_renderer+'.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
