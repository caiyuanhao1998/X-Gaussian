"""Train-only physical ray-consistency refinement; preserves X attributes."""
import argparse,csv,json,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm
from scene.gaussian_model import GaussianModel_Xray
from xct.data import CTData
from xct.field import render,query,configure_field
from xct.io import configure_from_ply,save_ct_ply
from xct.refinement import project_scales,prepare_scale_chart,finish_scale_chart
from xct.rays import project_volume
from xct.loss_utils import tv_3d_loss
from utils.loss_utils import ssim
from train_ct import evaluate


@torch.no_grad()
def ray_validation(data,pc,count=2048,method="siddon"):
    volume=query(pc,data.scanner['offOrigin'],data.scanner['nVoxel'],data.scanner['sVoxel'])['vol']
    errors=[]
    generator=torch.Generator().manual_seed(31415)
    for camera in data.test:
        indices=torch.randint(camera.image_width*camera.image_height,(count,),generator=generator).cuda()
        pred=project_volume(volume,camera,data.scanner,indices,method=method)
        target=camera.original_image.flatten()[indices]
        errors.append(((pred-target)**2).mean())
    return float((-10*torch.log10(torch.stack(errors))).mean())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model_path',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--source_path',help='Override the dataset path saved in config')
    p.add_argument('--checkpoint',type=int,default=5000)
    p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--ray_weight',type=float,default=1.)
    p.add_argument('--ray_count',type=int,default=2048)
    p.add_argument('--ray_samples',type=int,default=384)
    p.add_argument('--ray_method',choices=['siddon','trilinear'],default='siddon')
    p.add_argument('--protocol',choices=['trainval','test'],default='trainval')
    p.add_argument('--evaluate_initial',action='store_true')
    p.add_argument('--skip_evaluation',action='store_true')
    p.add_argument('--backend',choices=['standard','precision'])
    a=p.parse_args();random.seed(0);np.random.seed(0);torch.manual_seed(0)
    source=Path(a.model_path);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source/'config.json').read_text())
    if a.source_path:cfg['source_path']=a.source_path
    base_steps=cfg.get('total_iterations',a.checkpoint) if cfg.get('finetune') else a.checkpoint
    cfg.update(total_iterations=base_steps+a.iterations,source_model_path=str(source.resolve()),
               source_checkpoint=a.checkpoint,model_path=str(out),iterations=a.iterations,
               finetune='voxel ray consistency',ray_weight=a.ray_weight,ray_method=a.ray_method,
               ray_count=a.ray_count,ray_samples=a.ray_samples,protocol=a.protocol,eval_protocol=a.protocol)
    if a.backend is not None:cfg['backend']=a.backend
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    data=CTData(cfg['source_path'],load_test=a.protocol=='test' and not a.skip_evaluation)
    if a.protocol=='trainval':
        data.test=[c for i,c in enumerate(data.train) if i%5==0]
        data.train=[c for i,c in enumerate(data.train) if i%5!=0]
    pc=GaussianModel_Xray(cfg['sh_degree'])
    model=source/'point_cloud'/f'iteration_{a.checkpoint}'/'point_cloud.ply'
    pc.load_ply(str(model));configure_from_ply(pc,model,cfg)
    if a.backend is not None:configure_field(pc,pc.ct_activation,pc.ct_beta,a.backend)
    pc.spatial_lr_scale=1.
    options=SimpleNamespace(**cfg);pc.training_setup(options)
    rates={'xyz':1e-5,'f_dc':.0001,'f_rest':.000005,'opacity':.0003,'scaling':.00015,'rotation':.00005}
    if a.evaluate_initial:
        metrics=evaluate(data,pc,out,0,include_volume=False)
        metrics['ray_psnr_2d']=ray_validation(data,pc,method=a.ray_method)
        (out/'initial_validation.json').write_text(json.dumps(metrics,indent=2));print('INITIAL '+json.dumps(metrics),flush=True)
    with open(out/'loss.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['iteration','loss','l1','dssim','tv','ray']);writer.writeheader()
        stack=[];start=time.perf_counter()
        tv_n=np.array([32]*3);tv_s=torch.tensor(data.scanner['dVoxel']*tv_n,dtype=torch.float32)
        for step in tqdm(range(1,a.iterations+1),desc='physical ray refinement'):
            decay=.2**(step/a.iterations)
            for group in pc.optimizer.param_groups:group['lr']=rates[group['name']]*decay
            if not stack:stack=data.train.copy()
            cam=stack.pop(random.randrange(len(stack)))
            image=render(cam,pc)['render'];target=cam.original_image
            l1=(image-target).abs().mean();dssim=1-ssim(image,target)
            center=data.bbox[0]+tv_s/2+(data.bbox[1]-data.bbox[0]-tv_s)*torch.rand(3)
            tv=tv_3d_loss(query(pc,center,tv_n,tv_s,center_grad=cfg.get('tv_center_grad',1.))['vol'],reduction='mean')
            ray=l1.new_zeros(())
            if a.ray_weight:
                vol=query(pc,data.scanner['offOrigin'],data.scanner['nVoxel'],data.scanner['sVoxel'])['vol']
                indices=torch.randint(cam.image_width*cam.image_height,(a.ray_count,),device='cuda')
                ray=(project_volume(vol,cam,data.scanner,indices,a.ray_samples,method=a.ray_method)-target.flatten()[indices]).abs().mean()
            loss=.8*l1+.2*dssim+.05*tv+a.ray_weight*ray
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refinement loss')
            loss.backward();scale_state=prepare_scale_chart(pc,options);pc.optimizer.step();finish_scale_chart(pc,options,scale_state);pc.optimizer.zero_grad(set_to_none=True);project_scales(pc,options)
            writer.writerow(dict(iteration=step,loss=float(loss),l1=float(l1),dssim=float(dssim),tv=float(tv),ray=float(ray)))
        f.flush()
    if a.skip_evaluation:
        metrics=dict(iteration=a.iterations,points=len(pc.get_xyz),evaluation_deferred=True,wall_seconds=time.perf_counter()-start)
    else:
        metrics=evaluate(data,pc,out,a.iterations,include_volume=a.protocol=='test',save_volume=a.protocol=='test')
        metrics.update(ray_psnr_2d=ray_validation(data,pc,method=a.ray_method),wall_seconds=time.perf_counter()-start)
    (out/'results.json').write_text(json.dumps(metrics,indent=2))
    save_ct_ply(pc,out/'point_cloud'/f'iteration_{a.iterations}'/'point_cloud.ply')
    print('REFINEMENT COMPLETE '+json.dumps(metrics),flush=True)


if __name__=='__main__':main()
