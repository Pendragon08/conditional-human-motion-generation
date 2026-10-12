import pathlib,json,numpy as np
p=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/task_matched_feedback_v2_20261012/raw/val_shared_s11_r0')
print('META_KEYS',list(json.loads((p/'metadata.json').read_text())))
a=np.load(p/'trajectories.npz',allow_pickle=False)
print('ARRAYS',{k:list(a[k].shape) for k in a.files})
