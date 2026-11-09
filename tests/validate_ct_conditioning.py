"""Effective latent Adam conditioning and mass conservation, without new attributes."""
import sys,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from types import SimpleNamespace
from validate_ct_stable import prepared
from xct.field import configure_field,attenuation,radiative_channels,query
from xct.refinement import inverse_softplus,update_stable_rates,refine
pc=configure_field(prepared(),'softplus_sum',1.)
with torch.no_grad():
 pc._opacity[:]=inverse_softplus(torch.tensor([[.2]],device='cuda'))-radiative_channels(pc).mean(1,keepdim=True)
latent=torch.nn.Parameter(inverse_softplus(torch.tensor([[.2]],device='cuda')))
reference=torch.optim.Adam([latent],lr=.01,eps=1e-15)
args=SimpleNamespace(position_lr_max_steps=30000,feature_lr=.002,opacity_lr=.006,scaling_lr=.005,rotation_lr=.001,lr_final_ratio=.1,density_lr_init=.01,density_lr_final=.001)
for group in pc.optimizer.param_groups:
 if group['name'] in ['xyz','scaling','rotation']:group['lr']=0
for step in range(1,101):
 update_stable_rates(pc,args,step)
 for group in pc.optimizer.param_groups:
  if group['name'] in ['xyz','scaling','rotation']:group['lr']=0
 reference.param_groups[0]['lr']=.01*(.1**(step/30000))
 loss=(attenuation(pc)-.15).square().sum();loss.backward();pc.optimizer.step();pc.optimizer.zero_grad(set_to_none=True)
 other=(torch.nn.functional.softplus(latent)-.15).square().sum();other.backward();reference.step();reference.zero_grad(set_to_none=True)
 assert torch.allclose(attenuation(pc),torch.nn.functional.softplus(latent),atol=3e-7,rtol=2e-5)
print('PASS six-group X optimization matches scalar effective-density Adam for 100 steps')
for split in [False,True]:
 pc=configure_field(prepared(),'softplus_sum',1.)
 opts=SimpleNamespace(percent_dense=.001 if split else 1.,densify_grad_threshold=.1,max_points=100,scale_floor=.001,scale_ceiling=1.,current_step=1000,prune_from=2000,density_min=1e-6)
 before=query(pc,[0,0,0],[17]*3,[1.2]*3)['vol']
 stats=refine(pc,opts,torch.tensor([[-2.,-2.,-2.],[2.,2.,2.]]))
 assert abs(stats['mass_before']/stats['mass_after_split']-1)<3e-6
 if not split:assert torch.allclose(before,query(pc,[0,0,0],[17]*3,[1.2]*3)['vol'],rtol=1e-5,atol=2e-6)
 print('PASS conditioned head mass conservation','split' if split else 'clone')
pc=configure_field(prepared(),'softplus_sum',1.)
a=query(pc,[0,0,0],[17]*3,[1.2]*3,center_grad=0)['vol'].sum()
g=torch.autograd.grad(a,[pc._xyz,pc._features_dc],allow_unused=True)
assert (g[0] is None or not g[0].abs().any()) and g[1].abs().sum()>0
print('PASS optional TV center gradient isolation retains density gradients')
from xct.refinement import prepare_scale_chart,finish_scale_chart
pc=configure_field(prepared(),'softplus_sum',1.)
with torch.no_grad():pc._scaling.fill_(math.log(.01))
scale_ref=torch.nn.Parameter(torch.logit(torch.tensor([[.009/.999]*3],device='cuda')))
ref_opt=torch.optim.Adam([scale_ref],lr=.005,eps=1e-15)
args.bounded_scale_optimizer=True;args.scale_floor=.001;args.scale_ceiling=1.
for group in pc.optimizer.param_groups:group['lr']=.005 if group['name']=='scaling' else 0.
for step in range(100):
    loss=((pc.get_scaling-.02)**2).sum();loss.backward();state=prepare_scale_chart(pc,args);pc.optimizer.step();finish_scale_chart(pc,args,state);pc.optimizer.zero_grad(set_to_none=True)
    sig=.001+.999*torch.sigmoid(scale_ref);((sig-.02)**2).sum().backward();ref_opt.step();ref_opt.zero_grad(set_to_none=True)
    assert torch.allclose(pc.get_scaling,.001+.999*torch.sigmoid(scale_ref),rtol=1e-5,atol=2e-7)
print('PASS bounded-coordinate Adam matches reference while retaining X log-scale attributes')
