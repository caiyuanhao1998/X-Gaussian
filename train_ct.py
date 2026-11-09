"""Train quantitative CT using the unchanged X-Gaussian point attributes.

See CT_RECONSTRUCTION.md for the forward model and upstream CUDA attribution.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel_Xray
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import l1_loss, ssim
from xct.data import CTData
from xct.io import save_ct_ply
from xct.field import attenuation, render, query, configure_field
from xct.refinement import inverse_softplus, refine, update_stable_rates, project_scales, prepare_scale_chart, finish_scale_chart
from xct.loss_utils import tv_3d_loss
from xct.metrics import metric_proj, metric_vol


def initialize(data, degree, activation="relu", beta=1., points=None, backend="standard"):
    points = data.initialization() if points is None else points
    pc = GaussianModel_Xray(degree)
    # Retain the original raw opacity, 48 radiative features, log scales and quaternion.
    # Invert the chosen CT head so the initial physical density equals the FDK density.
    colors = np.repeat(points[:,3:4]/0.1,3,axis=1)
    if activation=='softplus':
        colors=inverse_softplus(torch.from_numpy(colors),beta).numpy()
    elif activation=='softplus_sum':
        latent=inverse_softplus(torch.from_numpy(points[:,3:4]),beta).numpy()
        colors=np.repeat(latent-np.log(.1/.9),3,axis=1)
    pc.create_from_pcd(BasicPointCloud(points[:,:3],colors,np.zeros_like(points[:,:3])),1.)
    return configure_field(pc,activation,beta,backend)


@torch.no_grad()
def evaluate(data, pc, out, step, save_volume=False, include_volume=True):
    out = Path(out)/'eval'/f'iter_{step:06d}'
    out.mkdir(parents=True,exist_ok=True)
    images = torch.cat([render(cam,pc)['render'] for cam in data.test],0).permute(1,2,0)
    gt = torch.cat([cam.original_image for cam in data.test],0).permute(1,2,0)
    if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite projection')
    metrics = dict(iteration=step,points=len(pc.get_xyz),
        psnr_2d=metric_proj(gt,images,'psnr')[0],ssim_2d=metric_proj(gt,images,'ssim')[0],
        raw_psnr_2d=float((-10*torch.log10(((gt-images)**2).mean((0,1)))).mean()))
    if include_volume:
        volume = query(pc,data.scanner['offOrigin'],data.scanner['nVoxel'],data.scanner['sVoxel'])['vol']
        vol_gt=data.load_ground_truth()
        if not torch.isfinite(volume).all():raise FloatingPointError('Nonfinite volume')
        metrics.update(psnr_3d=metric_vol(vol_gt,volume,'psnr')[0],ssim_3d=metric_vol(vol_gt,volume,'ssim')[0])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    if save_volume and include_volume:
        np.save(out/'vol_pred.npy',volume.cpu().numpy())
        np.save(out/'vol_gt.npy',vol_gt.cpu().numpy())
        np.savez(out/'projections.npz',pred=images.cpu().numpy(),gt=gt.cpu().numpy())
    print('CT EVAL '+json.dumps(metrics),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser()
    OptimizationParams(parser)
    parser.add_argument('--config',default='config/ct_teapot.yaml')
    parser.add_argument('--source_path')
    parser.add_argument('--model_path')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--sh_degree',type=int,default=3)
    parser.add_argument('--lambda_tv',type=float,default=.05)
    parser.add_argument('--tv_vol_size',type=int,default=32)
    parser.add_argument('--max_points',type=int,default=500000)
    parser.add_argument('--eval_every',type=int,default=5000)
    parser.add_argument('--checkpoint_every',type=int,default=10000)
    parser.add_argument('--resume')
    parser.add_argument('--profile',choices=['legacy','stable'],default='legacy')
    parser.add_argument('--activation',choices=['relu','softplus','softplus_sum'],default='relu')
    parser.add_argument('--beta',type=float,default=1.)
    parser.add_argument('--backend',choices=['standard','precision'],default='standard')
    parser.add_argument('--scale_floor',type=float,default=.001)
    parser.add_argument('--scale_ceiling',type=float,default=1.)
    parser.add_argument('--lr_final_ratio',type=float,default=.1)
    parser.add_argument('--density_min',type=float,default=1e-6)
    parser.add_argument('--prune_from',type=int,default=2000)
    parser.add_argument('--eval_protocol',choices=['test','trainval'],default='test')
    parser.add_argument('--init_path')
    parser.add_argument('--final_eval_only',action='store_true')
    parser.add_argument('--full_features',action='store_true')
    parser.add_argument('--density_lr_init',type=float,default=.01)
    parser.add_argument('--density_lr_final',type=float,default=.001)
    parser.add_argument('--tv_center_grad',type=float,default=1.)
    parser.add_argument('--bounded_scale_optimizer',action='store_true')
    parser.add_argument('--dropout_initial',type=float,default=0.)
    parser.add_argument('--dropout_until',type=int,default=10000)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    # Explicit CLI values override the config; argparse defaults do not.
    import sys
    for k,v in cfg.items():
        if not any(t == '--'+k or t.startswith('--'+k+'=') for t in sys.argv[1:]):
            setattr(args,k,v)
    if not 0<=args.dropout_initial<1:raise ValueError('dropout_initial must lie in [0,1)')
    if args.dropout_initial and args.dropout_until<1:raise ValueError('dropout_until must be positive')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    out=Path(args.model_path);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    data=CTData(args.source_path,load_test=args.eval_protocol=="test")
    if args.eval_protocol=='trainval':
        if not args.init_path:raise ValueError('Train-val requires a training-only FDK initialization')
        validation=[c for i,c in enumerate(data.train) if i%5==0]
        data.train=[c for i,c in enumerate(data.train) if i%5!=0]
        data.test=validation
        (out/'split.json').write_text(json.dumps(dict(training=[c.image_name for c in data.train],validation=[c.image_name for c in data.test],official_test_used=False,volume_ground_truth_used=False),indent=2))
    points=np.load(args.init_path) if args.init_path else None
    pc=initialize(data,args.sh_degree,args.activation,args.beta,points,args.backend)
    if args.bounded_scale_optimizer:
        with torch.no_grad():pc._scaling.copy_(pc.get_scaling.clamp(args.scale_floor+1e-5,args.scale_ceiling-1e-5).log())
    if args.full_features:pc.active_sh_degree=pc.max_sh_degree
    if args.profile=='stable':project_scales(pc,args)
    pc.training_setup(args)
    refinement_log=open(out/'refinement.jsonl','a' if args.resume else 'w')
    first=0
    if args.resume:
        params,first=torch.load(args.resume)
        pc.restore(params,args)
    writer_file=open(out/'loss.csv','a' if args.resume else 'w',newline='')
    writer=csv.DictWriter(writer_file,fieldnames=['iteration','loss','l1','dssim','tv','points'])
    if not args.resume:writer.writeheader()
    started=time.perf_counter();stack=[]
    progress=tqdm(range(first+1,args.iterations+1),desc='X-Gaussian CT')
    tv_n=np.array([args.tv_vol_size]*3)
    tv_s=torch.tensor(data.scanner['dVoxel']*tv_n,dtype=torch.float32)
    for step in progress:
        pc.update_learning_rate(step)
        if args.profile=="stable":update_stable_rates(pc,args,step)
        if step%1000==0:pc.oneupSHdegree()
        if not stack:stack=data.train.copy()
        cam=stack.pop(random.randrange(len(stack)))
        drop=args.dropout_initial*max(0.,1-step/args.dropout_until)
        mask=((torch.rand((len(pc.get_xyz),1),device='cuda')>=drop).float()/(1-drop)) if drop>0 else None
        pkg=render(cam,pc,attenuation_scale=mask)
        image=pkg['render'];gt=cam.original_image
        ll1=l1_loss(image,gt);dssim=1-ssim(image,gt)
        tv=ll1.new_zeros(())
        if args.lambda_tv>0:
            center=data.bbox[0]+tv_s/2+(data.bbox[1]-data.bbox[0]-tv_s)*torch.rand(3)
            tv=tv_3d_loss(query(pc,center,tv_n,tv_s,center_grad=args.tv_center_grad)['vol'],reduction='mean')
        loss=(1-args.lambda_dssim)*ll1+args.lambda_dssim*dssim+args.lambda_tv*tv
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward()
        with torch.no_grad():
            # Use the original X split/clone/prune machinery and optimizer groups.
            if step<args.densify_until_iter:
                visible=pkg['visibility_filter']
                pc.max_radii2D[visible]=torch.maximum(pc.max_radii2D[visible],pkg['radii'][visible])
                pc.add_densification_stats(pkg['viewspace_points'],visible)
                if step>args.densify_from_iter and step%args.densification_interval==0:
                    if args.profile=='stable':
                        args.current_step=step
                        stats=refine(pc,args,data.bbox)
                        refinement_log.write(json.dumps(dict(iteration=step,**stats))+'\n');refinement_log.flush()
                    else:
                        if len(pc.get_xyz)<args.max_points:
                            pc.densify_and_prune(args.densify_grad_threshold,0.,2.,None)
                        xyz=pc.get_xyz
                        bad=(attenuation(pc).squeeze(1)<1e-5)|((xyz<data.bbox[0].cuda())|(xyz>data.bbox[1].cuda())).any(1)
                        pc.prune_points(bad)
            if len(pc.get_xyz)==0:raise RuntimeError('All Gaussians pruned')
            if step%500==0:
                for group in pc.optimizer.param_groups:
                    g=group['params'][0].grad
                    if g is not None and not torch.isfinite(g).all():
                        raise FloatingPointError(f'Nonfinite gradient: {group["name"]}')
            if step<args.iterations:
                scale_state=prepare_scale_chart(pc,args)
                pc.optimizer.step();finish_scale_chart(pc,args,scale_state);pc.optimizer.zero_grad(set_to_none=True)
                if args.profile=="stable":project_scales(pc,args)
            writer.writerow(dict(iteration=step,loss=float(loss),l1=float(ll1),dssim=float(dssim),tv=float(tv),points=len(pc.get_xyz)))
            if step%10==0:progress.set_postfix(loss=f'{loss.item():.4g}',pts=len(pc.get_xyz))
            if step%500==0:writer_file.flush()
            if (step==1 and not args.final_eval_only) or step%args.eval_every==0 or step==args.iterations:
                metrics=evaluate(data,pc,out,step,save_volume=step==args.iterations,include_volume=args.eval_protocol=="test")
            if step%args.checkpoint_every==0 or step==args.iterations:
                save_ct_ply(pc,out/'point_cloud'/f'iteration_{step}'/'point_cloud.ply')
                torch.save((pc.capture(),step),out/f'chkpnt{step}.pth')
    writer_file.close();refinement_log.close()
    summary=dict(metrics,wall_seconds=time.perf_counter()-started,
                 attributes=pc.construct_list_of_attributes(),source_path=str(data.path))
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Training complete. '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
