"""Train one Gaussian field for X-ray projection and direct CT reconstruction."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

from scene.gaussian_model import GaussianModel_Xray
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import l1_loss, ssim
from xct.data import CTData
from xct.field import configure_field, query, render
from xct.io import configure_from_ply, save_ct_ply
from xct.optimizers import TrackedAttenuationAdam, BoundedScaleAdam
from xct.loss_utils import tv_3d_loss
from xct.rays import padded_scanner, project_volume
from xct.refinement import AttenuationAdam, inverse_softplus, project_scales, refine, update_stable_rates


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize(data, degree, init_path=None):
    points = np.load(init_path) if init_path else data.initialization()
    if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all() or np.any(points[:,3]<=0):
        raise ValueError('Initialization must contain finite xyz and positive density columns')
    if data.cache_dir is not None:
        np.save(data.cache_dir/'initialization.npy',points)
    # Keep sigmoid opacity=0.1 and invert the radiative Softplus to match FDK density.
    colors = np.repeat(points[:, 3:4] / .1, 3, axis=1)
    colors = inverse_softplus(torch.from_numpy(colors)).numpy()
    pc = GaussianModel_Xray(degree)
    pc.create_from_pcd(BasicPointCloud(points[:, :3], colors, np.zeros_like(points[:, :3])), 1.)
    return configure_field(pc, 'softplus', 1.)


def save_model(pc, out, iteration):
    path = out / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
    save_ct_ply(pc, path)
    return path


def tv_loss(pc, data, args, center_grad=None):
    n = np.array([args.tv_vol_size] * 3)
    size = torch.tensor(data.scanner['dVoxel'] * n, dtype=torch.float32)
    center = data.bbox[0] + size / 2 + (data.bbox[1] - data.bbox[0] - size) * torch.rand(3)
    # TV trains attenuation/covariance; projection and physical-ray losses train centers.
    if center_grad is None:
        center_grad = getattr(args, 'tv_center_grad', 0.)
    volume = query(pc, center, n, size, center_grad=center_grad)['vol']
    return tv_3d_loss(volume, reduction='mean')


def train_base(data, args, out, writer):
    pc = initialize(data, args.sh_degree, getattr(args,'init_path',None))
    pc.volume_geometry = args.volume_geometry
    project_scales(pc, args)
    from xct.initialization import save_initialization
    save_initialization(pc, data, out, getattr(args,'init_path',None))
    pc.training_setup(args)
    conditioned = getattr(args, 'base_density_lr', 0.) > 0
    density = TrackedAttenuationAdam(pc, eps=getattr(args, 'base_density_eps', 1e-15)) if conditioned else None
    scaling = BoundedScaleAdam(pc, args.scale_floor, args.scale_ceiling) if conditioned and getattr(args, 'base_bounded_scale', True) else None
    started = time.time()
    topology = []
    stack = []
    progress = tqdm(range(1, args.iterations + 1), desc='Joint 2D/3D training')
    for step in progress:
        pc.update_learning_rate(step)
        update_stable_rates(pc, args, step)
        if step % 1000 == 0:
            pc.oneupSHdegree()
        if not stack:
            stack = data.train.copy()
        camera = stack.pop(random.randrange(len(stack)))
        package = render(camera, pc)
        l1 = l1_loss(package['render'], camera.original_image)
        dssim = 1 - ssim(package['render'], camera.original_image)
        tv = tv_loss(pc, data, args) if args.lambda_tv else l1.new_zeros(())
        loss = (1 - args.lambda_dssim) * l1 + args.lambda_dssim * dssim + args.lambda_tv * tv
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at iteration {step}')
        loss.backward()
        with torch.no_grad():
            changing = False
            if step < args.densify_until_iter:
                visible = package['visibility_filter']
                pc.max_radii2D[visible] = torch.maximum(pc.max_radii2D[visible], package['radii'][visible])
                pc.add_densification_stats(package['viewspace_points'], visible)
                if step > args.densify_from_iter and step % args.densification_interval == 0:
                    args.current_step = step
                    event = refine(pc, args, data.bbox)
                    event["iteration"] = step
                    topology.append(event)
                    changing = True
            if len(pc.get_xyz) == 0:
                raise RuntimeError('All Gaussians were pruned')
            if step % 500 == 0:
                for group in pc.optimizer.param_groups:
                    grad = group['params'][0].grad
                    if grad is not None and not torch.isfinite(grad).all():
                        raise FloatingPointError(f'Nonfinite gradient: {group["name"]}')
            if step < args.iterations and not (conditioned and changing):
                if density:
                    density.step(args.base_density_lr * args.lr_final_ratio ** (step / args.position_lr_max_steps))
                if scaling:
                    rate = next(g['lr'] for g in pc.optimizer.param_groups if g['name'] == 'scaling')
                    scaling.step(rate)
                pc.optimizer.step()
                pc.optimizer.zero_grad(set_to_none=True)
                project_scales(pc, args)
            if conditioned:
                pc.optimizer.zero_grad(set_to_none=True)
                project_scales(pc, args)
            writer.writerow(dict(iteration=step, stage='base', loss=float(loss), l1=float(l1),
                                 dssim=float(dssim), tv=float(tv), ray=0., points=len(pc.get_xyz)))
            if step % 10 == 0:
                progress.set_postfix(loss=f'{loss.item():.4g}', points=len(pc.get_xyz))
            if step % args.checkpoint_every == 0 or step == args.iterations:
                save_model(pc, out, step)
    (out / 'topology_events.json').write_text(json.dumps(topology, indent=2) + '\n')
    (out / 'base_timing.json').write_text(json.dumps(dict(train_seconds=time.time()-started,
        training_steps=args.iterations, training_projections=len(data.train), points=len(pc.get_xyz)), indent=2) + '\n')
    return out / 'point_cloud' / f'iteration_{args.iterations}' / 'point_cloud.ply'


def refine_volume(data, args, out, model, writer):
    # Reset RNG and Adam at the refinement boundary, matching the measured recipe.
    seed_all(args.seed)
    pc = GaussianModel_Xray(args.sh_degree)
    pc.load_ply(str(model))
    configure_from_ply(pc, model)
    pc.spatial_lr_scale = 1.
    pc.training_setup(args)
    feature_rate = getattr(args, 'refine_feature_lr', .0001)
    rates = dict(xyz=getattr(args, 'refine_position_lr', 1e-5), f_dc=feature_rate,
                 f_rest=getattr(args, 'refine_feature_rest_lr', feature_rate / 20),
                 opacity=getattr(args, 'refine_opacity_lr', .0003),
                 scaling=getattr(args, 'refine_scaling_lr', .00015),
                 rotation=getattr(args, 'refine_rotation_lr', .00005))
    final_ratio = getattr(args, 'refine_lr_final_ratio', .2)
    tv_weight = getattr(args, 'refine_lambda_tv', .05)
    dssim_weight = getattr(args, 'refine_lambda_dssim', .2)
    projection_weight = getattr(args, 'refine_projection_weight', 1.)
    ray_loss = getattr(args, 'refine_ray_loss', 'l1')
    density_rate = getattr(args, 'refine_density_lr', 0.)
    if density_rate < 0:
        raise ValueError('Attenuation learning rate must be nonnegative')
    density_optimizer = AttenuationAdam(pc, getattr(args, 'refine_density_eps', 1e-8)) if density_rate else None
    if density_optimizer:
        for name in ['f_dc', 'f_rest', 'opacity']:
            rates[name] = 0.
    if not 0 < final_ratio <= 1 or min(rates.values()) < 0:
        raise ValueError('Refinement rates must be nonnegative and final ratio in (0, 1]')
    if min(tv_weight, projection_weight, args.ray_weight) < 0 or not 0 <= dssim_weight <= 1:
        raise ValueError('Invalid refinement loss weights')
    if ray_loss not in ['l1', 'mse']:
        raise ValueError('refine_ray_loss must be l1 or mse')
    scale_args = args
    if hasattr(args, 'refine_scale_floor'):
        if not 0 < args.refine_scale_floor <= args.scale_ceiling:
            raise ValueError('Refinement scale floor must be positive and below the ceiling')
        scale_args = SimpleNamespace(**{**vars(args), 'scale_floor': args.refine_scale_floor})
        with torch.no_grad():
            project_scales(pc, scale_args)
    ray_scanner = padded_scanner(data.scanner, getattr(args, 'ray_padding', 0))
    stack = []
    progress = tqdm(range(1, args.refine_iterations + 1), desc='Joint projection/volume refinement')
    for step in progress:
        decay = final_ratio ** (step / args.refine_iterations)
        for group in pc.optimizer.param_groups:
            group['lr'] = rates[group['name']] * decay
        if not stack:
            stack = data.train.copy()
        camera = stack.pop(random.randrange(len(stack)))
        target = camera.original_image
        if projection_weight:
            image = render(camera, pc)['render']
            l1 = (image - target).abs().mean()
            dssim = 1 - ssim(image, target)
        else:
            l1 = dssim = pc.get_xyz.new_zeros(())
        tv = tv_loss(pc, data, args, getattr(args, 'refine_tv_center_grad', 0.)) if tv_weight else l1.new_zeros(())
        if args.ray_weight:
            volume = query(pc, ray_scanner['offOrigin'], ray_scanner['nVoxel'], ray_scanner['sVoxel'])['vol']
            indices = torch.randint(camera.image_width * camera.image_height, (args.ray_count,), device='cuda')
            residual = project_volume(volume, camera, ray_scanner, indices) - target.flatten()[indices]
            ray = residual.abs().mean() if ray_loss == 'l1' else residual.square().mean()
        else:
            ray = l1.new_zeros(())
        projection = (1 - dssim_weight) * l1 + dssim_weight * dssim
        if projection_weight != 1.:
            projection = projection_weight * projection
        loss = projection + tv_weight * tv + args.ray_weight * ray
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite refinement loss at iteration {step}')
        loss.backward()
        if density_optimizer:
            density_optimizer.step(density_rate * decay)
        pc.optimizer.step()
        pc.optimizer.zero_grad(set_to_none=True)
        project_scales(pc, scale_args)
        writer.writerow(dict(iteration=args.iterations + step, stage='refine', loss=float(loss),
                             l1=float(l1), dssim=float(dssim), tv=float(tv), ray=float(ray), points=len(pc.get_xyz)))
        if step % 10 == 0:
            progress.set_postfix(loss=f'{loss.item():.4g}')
        checkpoint_every = getattr(args, 'refine_checkpoint_every', 0)
        if checkpoint_every and step % checkpoint_every == 0 and step != args.refine_iterations:
            save_model(pc, out, args.iterations + step)
    return save_model(pc, out, args.iterations + args.refine_iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/synthetic/chest.yaml')
    parser.add_argument('-s', '--source_path')
    parser.add_argument('-m', '--model_path')
    parser.add_argument('--iterations', type=int, help='Base joint-training steps')
    parser.add_argument('--refine_iterations', type=int, help='Joint volume-ray refinement steps')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--init_path',help='Reuse a saved initialization.npy')
    cli = parser.parse_args()
    config = yaml.safe_load(Path(cli.config).read_text())
    for key, value in vars(cli).items():
        if value is not None:
            config[key] = value
    if cli.refine_iterations is not None:
        config.pop('refinement_stages', None)
    config.setdefault('refine_iterations', 0)
    args = SimpleNamespace(**config)
    stages = config.get('refinement_stages', [])
    if stages and any(not isinstance(stage.get('iterations'), int) or stage['iterations'] < 1 for stage in stages):
        parser.error('Each refinement stage needs a positive integer iterations value')
    extra_steps = sum(stage['iterations'] for stage in stages) if stages else args.refine_iterations
    if args.iterations < 1 or args.refine_iterations < 0 or args.checkpoint_every < 1 or args.ray_count < 1:
        parser.error('Training/checkpoint/ray counts must be positive; refinement steps may be zero')
    seed_all(args.seed)
    torch.cuda.set_device(0)
    out = Path(args.model_path)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        parser.error('Output already contains a run; choose another --model_path')
    data = CTData(args.source_path, load_test=False, cache_dir=out)
    args.volume_geometry = {k: np.asarray(data.scanner[k]).tolist() for k in ['nVoxel', 'sVoxel', 'offOrigin']}
    config.update(volume_geometry=args.volume_geometry, source_path=str(data.path), total_iterations=args.iterations + extra_steps,
                  activation='softplus', beta=1., backend='standard')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    fields = ['iteration', 'stage', 'loss', 'l1', 'dssim', 'tv', 'ray', 'points']
    with open(out / 'loss.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        model = train_base(data, args, out, writer)
        if stages:
            offset = args.iterations
            for stage in stages:
                settings = {**vars(args), **stage, 'iterations': offset, 'refine_iterations': stage['iterations']}
                stage_args = SimpleNamespace(**settings)
                model = refine_volume(data, stage_args, out, model, writer)
                offset += stage['iterations']
        elif args.refine_iterations:
            model = refine_volume(data, args, out, model, writer)
    (out / 'training_complete.json').write_text(json.dumps(dict(final_iteration=args.iterations + extra_steps,
        training_projections=len(data.train), model=str(model)), indent=2) + '\n')
    print(f'Training complete: {model}')


if __name__ == '__main__':
    main()
