"""Seeded shape/backprop witness for the paper-sized official MoE; not PPO training."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch,yaml

def main():
 p=argparse.ArgumentParser();p.add_argument('--official-source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);torch.manual_seed(42)
 spec=importlib.util.spec_from_file_location('official_phygile_policy',a.official_source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))['policy'];cfg.pop('class_name')
 policy=m.MoEActorCritic(616,748,29,**cfg)
 layers=[x for x in policy.experts[0].modules() if isinstance(x,torch.nn.Linear)]
 shapes=[[x.in_features,x.out_features] for x in layers]
 assert [x[1] for x in shapes]==[2048,1024,512,29],shapes
 rows=[]
 for active in [1,10]:
  policy.set_max_active_experts(active)
  actor=torch.randn(2,616);critic=torch.randn(2,748)
  actions=policy.act_inference(actor);value=policy.evaluate(critic)
  loss=actions.square().mean()+value.square().mean();policy.zero_grad(set_to_none=True);loss.backward()
  assert actions.shape==(2,29) and torch.isfinite(loss)
  assert all(torch.isfinite(x.grad).all() for x in policy.parameters() if x.grad is not None)
  rows.append({'active_experts':active,'actions_shape':list(actions.shape),'loss':float(loss.detach()),'finite_gradients':True})
 result={'status':'PASS','seed':42,'scope':'synthetic CPU shape and gradient witness only; zero PPO updates',
         'source_sha256':hashlib.sha256(a.official_source.read_bytes()).hexdigest(),
         'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
         'expert_linear_shapes':shapes,'parameters':sum(x.numel() for x in policy.parameters()),'cases':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
