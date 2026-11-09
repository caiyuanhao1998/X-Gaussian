"""Check positivity, exact clone invariance, split mass, and unchanged schema."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from types import SimpleNamespace
import torch
from validate_ct import single
from arguments import OptimizationParams
from xct.field import configure_field,radiative_features,attenuation,query
from xct.refinement import refine,inverse_softplus


def prepared():
    pc=configure_field(single(),'softplus',1.)
    p=argparse.ArgumentParser();OptimizationParams(p);a=p.parse_args([])
    pc.spatial_lr_scale=1.;pc.training_setup(a);pc.max_radii2D=torch.zeros(1,device='cuda')
    pc.xyz_gradient_accum.fill_(1);pc.denom.fill_(1)
    return pc


pc=prepared();names=pc.construct_list_of_attributes()
x=torch.tensor([1e-5,.0075,.1,1.,10.],device='cuda')
assert torch.allclose(torch.nn.functional.softplus(inverse_softplus(x)),x,rtol=2e-6)
with torch.no_grad():pc._features_dc.fill_(-20)
v=radiative_features(pc);v.sum().backward()
assert v.item()>0 and pc._features_dc.grad.abs().sum().item()>0
print('PASS positive radiative field has nonzero tail gradient')
a=SimpleNamespace(percent_dense=1.,densify_grad_threshold=.1,max_points=100,scale_floor=.001,scale_ceiling=1.,current_step=1000,prune_from=2000,density_min=1e-6)
bbox=torch.tensor([[-2.,-2.,-2.],[2.,2.,2.]])
pc=prepared();before=query(pc,[0,0,0],[17]*3,[1.2]*3)['vol'];stats=refine(pc,a,bbox);after=query(pc,[0,0,0],[17]*3,[1.2]*3)['vol']
assert len(pc.get_xyz)==2 and torch.allclose(before,after,rtol=1e-5,atol=2e-6)
assert abs(stats['mass_before']/stats['mass_after_split']-1)<2e-6
print('PASS clone preserves complete attenuation field and mass',stats)
pc=prepared();a.percent_dense=.001;stats=refine(pc,a,bbox)
assert len(pc.get_xyz)==2 and abs(stats['mass_before']/stats['mass_after_split']-1)<2e-6
assert pc.construct_list_of_attributes()==names
print('PASS split conserves integrated Gaussian mass and original PLY schema',stats)
pc=prepared()
with torch.no_grad():pc._scaling[0]=torch.tensor([.001,.02,.1],device='cuda').log()
stats=refine(pc,a,bbox)
assert abs(stats['mass_before']/stats['mass_after_split']-1)<2e-6
assert (pc.get_scaling>=.001-1e-8).all()
print('PASS split mass conservation when the scale floor is active',stats)
import tempfile
from scene.gaussian_model import GaussianModel_Xray
from xct.io import save_ct_ply,configure_from_ply,field_metadata
with tempfile.TemporaryDirectory() as tmp:
    pc.volume_geometry=dict(nVoxel=[128,128,64],sVoxel=[2.,2.,1.],offOrigin=[0.,0.,0.])
    path=Path(tmp)/'standalone.ply';save_ct_ply(pc,path)
    loaded=GaussianModel_Xray(3);loaded.load_ply(str(path));configure_from_ply(loaded,path)
    assert field_metadata(path)['activation']=='softplus'
    assert loaded.volume_geometry==pc.volume_geometry
    assert loaded.construct_list_of_attributes()==pc.construct_list_of_attributes()
    assert torch.allclose(attenuation(pc),attenuation(loaded))
print('PASS standalone PLY comments retain field mapping without new point attributes')
