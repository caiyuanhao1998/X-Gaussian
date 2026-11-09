"""Stable CT optimization without adding per-Gaussian attributes."""
import math
import torch
from xct.field import attenuation, radiative_channels, configure_field
from utils.general_utils import inverse_sigmoid


def inverse_softplus(x, beta=1.):
    y=x*beta
    return (y+torch.log(-torch.expm1(-y)))/beta


def rescaled_features(dc, rest, raw, factor, activation, beta):
    channels = torch.nn.functional.softplus(raw,beta=beta) if activation=='softplus' else raw.clamp_min(0)
    target=channels*factor
    newraw=inverse_softplus(target.clamp_min(1e-12),beta) if activation=='softplus' else target
    # A DC shift preserves all higher-order radiative coefficients.
    from utils.sh_utils import C0
    return dc+(newraw-raw)[:,None,:]/C0,rest


@torch.no_grad()
def refine(pc, args, bbox):
    """Clone/split with conserved attenuation mass, then account for pruning."""
    before=len(pc.get_xyz)
    mass_before=float((attenuation(pc).squeeze(1)*pc.get_scaling.prod(1)).sum())
    grads=(pc.xyz_gradient_accum/pc.denom).nan_to_num(0.).squeeze(1)
    large=pc.get_scaling.amax(1)>args.percent_dense*2.
    eligible=grads>=args.densify_grad_threshold
    clone=eligible&~large
    split=eligible&large
    available=max(0,args.max_points-before)
    candidates=torch.where(clone|split)[0]
    if len(candidates)>available:
        keep=candidates[torch.argsort(grads[candidates],descending=True)[:available]]
        eligible=torch.zeros_like(eligible);eligible[keep]=True
        clone&=eligible;split&=eligible
    nclone=int(clone.sum());nsplit=int(split.sum())
    chunks=[]
    if nclone:
        # Both parent and colocated child get half the amplitude: identical field.
        if pc.ct_activation=='softplus_sum':
            latent=pc._opacity[clone]+radiative_channels(pc)[clone].mean(1,keepdim=True)
            half=pc._opacity[clone]+inverse_softplus(attenuation(pc)[clone]*.5,pc.ct_beta)-latent
        else:half=inverse_sigmoid(pc.get_opacity[clone]*.5)
        pc._opacity[clone]=half
        chunks.append([pc._xyz[clone],pc._features_dc[clone],pc._features_rest[clone],half,pc._scaling[clone],pc._rotation[clone]])
    if nsplit:
        scale=pc.get_scaling[split]
        newscale=(scale/1.6).clamp(args.scale_floor,args.scale_ceiling)
        raw=radiative_channels(pc)[split]
        factor=(scale.prod(1)/(2*newscale.prod(1)))[:,None]
        if pc.ct_activation=='softplus_sum':
            latent=pc._opacity[split]+raw.mean(1,keepdim=True)
            new_opacity=pc._opacity[split]+inverse_softplus(attenuation(pc)[split]*factor,pc.ct_beta)-latent
            dc,rest=pc._features_dc[split],pc._features_rest[split]
        else:
            dc,rest=rescaled_features(pc._features_dc[split],pc._features_rest[split],raw,factor,pc.ct_activation,pc.ct_beta)
            new_opacity=pc._opacity[split]
        from utils.general_utils import build_rotation
        std=scale.repeat(2,1)
        delta=torch.bmm(build_rotation(pc._rotation[split]).repeat(2,1,1),torch.normal(torch.zeros_like(std),std).unsqueeze(-1)).squeeze(-1)
        chunks.append([pc._xyz[split].repeat(2,1)+delta,dc.repeat(2,1,1),rest.repeat(2,1,1),new_opacity.repeat(2,1),newscale.log().repeat(2,1),pc._rotation[split].repeat(2,1)])
    if chunks:
        appended=[torch.cat([c[k] for c in chunks],0) for k in range(6)]
        pc.densification_postfix(*appended)
        remove=torch.cat([split,torch.zeros(len(pc.get_xyz)-before,dtype=torch.bool,device='cuda')])
        pc.prune_points(remove)
    mass_after_split=float((attenuation(pc).squeeze(1)*pc.get_scaling.prod(1)).sum())
    density=attenuation(pc).squeeze(1)
    low=density<args.density_min
    # Keep a weak point until the grace period ends; prune by actual attenuation.
    if args.current_step<args.prune_from:low=torch.zeros_like(low)
    xyz=pc.get_xyz;radius=3*pc.get_scaling.amax(1,keepdim=True)
    outside=((xyz+radius<bbox[0].cuda())|(xyz-radius>bbox[1].cuda())).any(1)
    nlow=int(low.sum());nout=int((outside&~low).sum())
    pc.prune_points(low|outside)
    pc.xyz_gradient_accum.zero_();pc.denom.zero_();pc.max_radii2D.zero_()
    return dict(before=before,cloned=nclone,split=nsplit,pruned_density=nlow,pruned_outside=nout,after=len(pc.get_xyz),mass_before=mass_before,mass_after_split=mass_after_split)


@torch.no_grad()
def update_stable_rates(pc,args,step):
    fraction=min(step/args.position_lr_max_steps,1.)
    for group in pc.optimizer.param_groups:
        initial={'f_dc':args.feature_lr,'f_rest':args.feature_lr/20,'opacity':args.opacity_lr,'scaling':args.scaling_lr,'rotation':args.rotation_lr}.get(group['name'])
        if initial is not None:group['lr']=initial*(args.lr_final_ratio**fraction)
    if getattr(pc,'ct_activation','relu')=='softplus_sum':
        from utils.sh_utils import C0,C1,C2,C3
        count=(pc.active_sh_degree+1)**2
        weights=[C0,-C1,C1,-C1]+C2+C3
        rest=sum(abs(x) for x in weights[1:count])
        total=args.density_lr_init*((args.density_lr_final/args.density_lr_init)**fraction)
        for group in pc.optimizer.param_groups:
            if group['name']=='opacity':group['lr']=total*.5
            elif group['name']=='f_dc':group['lr']=total*(.25 if rest else .5)/C0
            elif group['name']=='f_rest':group['lr']=total*.25/rest if rest else 0.



@torch.no_grad()
def project_scales(pc,args):
    pc._scaling.clamp_(math.log(args.scale_floor),math.log(args.scale_ceiling))


@torch.no_grad()
def prepare_scale_chart(pc,args):
    """Optimize in a bounded virtual coordinate while storing original X log scales."""
    if not getattr(args,'bounded_scale_optimizer',False) or pc._scaling.grad is None:return None
    old=pc._scaling.detach().clone();sigma=old.exp()
    t=((sigma-args.scale_floor)/(args.scale_ceiling-args.scale_floor)).clamp(1e-8,1-1e-7)
    derivative=(args.scale_ceiling-args.scale_floor)*t*(1-t)
    pc._scaling.grad.mul_(derivative/sigma)
    return old,torch.logit(t)


@torch.no_grad()
def finish_scale_chart(pc,args,state):
    if state is None:return
    old,virtual=state
    increment=pc._scaling-old
    sigma=args.scale_floor+(args.scale_ceiling-args.scale_floor)*torch.sigmoid(virtual+increment)
    pc._scaling.copy_(sigma.log())
