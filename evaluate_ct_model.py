"""Evaluate a frozen exported CT model without any further optimization."""
import argparse,json
from pathlib import Path
from scene.gaussian_model import GaussianModel_Xray
from xct.data import CTData
from xct.io import configure_from_ply
from train_ct import evaluate
p=argparse.ArgumentParser();p.add_argument('--model_path',required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--source_path',help='Override the dataset path saved in config');a=p.parse_args()
out=Path(a.model_path);cfg=json.loads((out/'config.json').read_text())
model=out/'point_cloud'/f'iteration_{a.iteration}'/'point_cloud.ply'
pc=GaussianModel_Xray(cfg['sh_degree']);pc.load_ply(str(model));configure_from_ply(pc,model,cfg)
data=CTData(a.source_path or cfg['source_path'])
result=evaluate(data,pc,out,a.iteration,save_volume=True)
prior=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else {}
prior.update(result);prior['evaluation_deferred']=False;prior['total_iterations']=cfg.get('total_iterations',cfg['iterations'])
(out/'results.json').write_text(json.dumps(prior,indent=2))
