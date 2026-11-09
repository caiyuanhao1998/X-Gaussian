"""Reload both models; evaluate identical original-resolution projections and volumes."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scene.gaussian_model import GaussianModel_Xray
from xct.data import CTData
from xct.io import configure_from_ply
from xct.field import render,query,attenuation,configure_field
from xct.metrics import metric_proj,metric_vol


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model_path',required=True)
    p.add_argument('--iteration',type=int,default=30000)
    p.add_argument('--source_path',help='Override the dataset path saved in config')
    p.add_argument('--baseline',required=True)
    p.add_argument('--r2_repo',type=Path,help='Optional path to an R2-Gaussian checkout')
    a=p.parse_args()
    out=Path(a.model_path).resolve();cfg=json.loads((out/'config.json').read_text())
    data=CTData(a.source_path or cfg['source_path'])
    pc=GaussianModel_Xray(cfg['sh_degree'])
    pc.load_ply(str(out/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply'))
    configure_from_ply(pc,out/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply',cfg)
    if a.r2_repo:sys.path.insert(0,str(a.r2_repo.resolve()))
    from r2_gaussian.gaussian import GaussianModel,render as render_r2
    from types import SimpleNamespace
    baseline=GaussianModel(None)
    baseline.load_ply(str(Path(a.baseline)/'point_cloud/iteration_30000/point_cloud.pickle'))
    baseline_vol=np.load(Path(a.baseline)/'point_cloud/iteration_30000/vol_pred.npy')
    vol_gt=data.load_ground_truth()
    result={};samples={}
    with torch.no_grad():
        gt=torch.cat([c.original_image for c in data.test],0).permute(1,2,0)
        for name,model,renderfun in [
            ('X-Gaussian-CT',pc,lambda c:render(c,pc)['render']),
            ('R2-Gaussian',baseline,lambda c:render_r2(c,baseline,SimpleNamespace(compute_cov3D_python=False,debug=False))['render'])]:
            image=torch.cat([renderfun(c) for c in data.test],0).permute(1,2,0)
            if name=='X-Gaussian-CT':
                volume=query(pc,data.scanner['offOrigin'],data.scanner['nVoxel'],data.scanner['sVoxel'])['vol']
                stored=np.load(out/'eval'/f'iter_{a.iteration:06d}'/'vol_pred.npy')
                assert np.allclose(volume.cpu().numpy(),stored,atol=2e-5,rtol=2e-5)
            else:volume=torch.from_numpy(baseline_vol).cuda()
            assert torch.isfinite(image).all() and torch.isfinite(volume).all()
            metrics=dict(psnr_2d=metric_proj(gt,image,'psnr')[0],ssim_2d=metric_proj(gt,image,'ssim')[0],
                raw_psnr_2d=float((-10*torch.log10(((gt-image)**2).mean((0,1)))).mean()),
                psnr_3d=metric_vol(vol_gt,volume,'psnr')[0],ssim_3d=metric_vol(vol_gt,volume,'ssim')[0],
                gaussians=len(model.get_xyz),test_views=len(data.test),
                resolution_2d=list(gt.shape[:2]),resolution_3d=list(volume.shape))
            result[name]=metrics
            samples[name+'_projection']=image[:,:,[0,25,50,75,99]].cpu().numpy()
            samples[name+'_slices']=np.stack([np.take(volume.cpu().numpy(),128,axis=k) for k in range(3)])
        samples['gt_projection']=gt[:,:,[0,25,50,75,99]].cpu().numpy()
        samples['gt_slices']=np.stack([np.take(vol_gt.cpu().numpy(),128,axis=k) for k in range(3)])
    result['delta_X_minus_R2']={k:result['X-Gaussian-CT'][k]-result['R2-Gaussian'][k] for k in ['psnr_2d','ssim_2d','raw_psnr_2d','psnr_3d','ssim_3d']}
    result['protocol']={'same_data':str(data.path),'resampling':False,'metric_implementation':'R2 official metric_proj/metric_vol; per-projection max normalization plus separate raw-intensity PSNR','ground_truth_used_for_training':False,'reloaded_X_PLY_matches_saved_volume':True,'baseline_retrained':False}
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(out/'comparison_samples.npz',**samples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,5,figsize=(13,8))
    for row,key in enumerate(['gt_projection','X-Gaussian-CT_projection','R2-Gaussian_projection']):
        for col in range(5):
            im=samples[key][...,col]
            axes[row,col].imshow(im,cmap='gray',vmin=0,vmax=samples['gt_projection'][...,col].max())
            axes[row,col].set_title(f'{key.split("_")[0]} / view {[0,25,50,75,99][col]}');axes[row,col].axis('off')
    fig.tight_layout();fig.savefig(out/'projection_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(5,3,figsize=(10,15))
    for axis in range(3):
        for row,key in enumerate(['gt_slices','X-Gaussian-CT_slices','R2-Gaussian_slices']):
            axes[row,axis].imshow(samples[key][axis],cmap='gray',vmin=0,vmax=1)
            axes[row,axis].set_title(f'{key.split("_")[0]} / axis {axis}');axes[row,axis].axis('off')
        for row,key in [(3,'X-Gaussian-CT_slices'),(4,'R2-Gaussian_slices')]:
            axes[row,axis].imshow(np.abs(samples[key][axis]-samples['gt_slices'][axis]),cmap='magma',vmin=0,vmax=.05)
            axes[row,axis].set_title(f'{key.split("_")[0]} absolute error');axes[row,axis].axis('off')
    fig.tight_layout();fig.savefig(out/'volume_comparison.png',dpi=160);plt.close(fig)
    import csv
    with open(out/'loss.csv') as f:loss=list(csv.DictReader(f))
    values=np.array([float(x['loss']) for x in loss]);assert np.isfinite(values).all() and len(values)==cfg['iterations']
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(np.arange(len(values)-99)+100,np.convolve(values,np.ones(100)/100,'valid'))
    ax.set(xlabel='Iteration',ylabel='Total loss (100-step mean)',yscale='log',title='X-Gaussian CT / original-resolution teapot')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'loss_curve.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
