"""Conditioned updates of existing Gaussian features and log scales."""
import torch
from xct.refinement import AttenuationAdam

def optimizer_state(pc,parameter,kind):
    state=pc.optimizer.state[parameter]
    if not state:
        state.update(step=torch.tensor(0.),exp_avg=torch.zeros_like(parameter),
                     exp_avg_sq=torch.zeros_like(parameter),conditioning=kind)
    if state.get('conditioning')!=kind:raise RuntimeError('Optimizer state uses a different parameterization')
    return state

class TrackedAttenuationAdam(AttenuationAdam):
    @torch.no_grad()
    def step(self,rate):
        state=optimizer_state(self.pc,self.pc._features_dc,'attenuation')
        self.m=state['exp_avg'][:,0,:1]
        self.v=state['exp_avg_sq'][:,0,:1]
        self.steps=int(state['step'].item())
        super().step(rate)
        state['step'].fill_(self.steps)

class BoundedScaleAdam:
    def __init__(self,pc,low,high,eps=1e-15):
        if not 0<low<high:raise ValueError('Invalid scale bounds')
        self.pc,self.low,self.high,self.eps=pc,low,high,eps
    @torch.no_grad()
    def step(self,rate):
        pc=self.pc;scale=pc.get_scaling
        u=((scale-self.low)/(self.high-self.low)).clamp(1e-7,1-1e-7)
        z=torch.logit(u)
        gradient=pc._scaling.grad*(self.high-self.low)*u*(1-u)/scale
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite bounded-scale gradient')
        state=optimizer_state(pc,pc._scaling,'bounded_scale')
        state['step'].add_(1);step=int(state['step'].item())
        state['exp_avg'].mul_(.9).add_(gradient,alpha=.1)
        state['exp_avg_sq'].mul_(.999).addcmul_(gradient,gradient,value=.001)
        update=(state['exp_avg']/(1-.9**step))/((state['exp_avg_sq']/(1-.999**step)).sqrt()+self.eps)
        pc._scaling.copy_((self.low+(self.high-self.low)*torch.sigmoid(z-rate*update)).log())
        pc._scaling.grad=None
