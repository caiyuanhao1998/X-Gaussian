"""Stable CT optimization without adding per-Gaussian attributes."""
import math
import torch
from xct.field import attenuation, radiative_channels
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
        half=inverse_sigmoid(pc.get_opacity[clone]*.5)
        pc._opacity[clone]=half
        chunks.append([pc._xyz[clone],pc._features_dc[clone],pc._features_rest[clone],half,pc._scaling[clone],pc._rotation[clone]])
    if nsplit:
        scale=pc.get_scaling[split]
        newscale=(scale/1.6).clamp(args.scale_floor,args.scale_ceiling)
        raw=radiative_channels(pc)[split]
        factor=(scale.prod(1)/(2*newscale.prod(1)))[:,None]
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


@torch.no_grad()
def project_scales(pc,args):
    pc._scaling.clamp_(math.log(args.scale_floor),math.log(args.scale_ceiling))
