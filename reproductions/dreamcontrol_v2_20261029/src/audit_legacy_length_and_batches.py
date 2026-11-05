"""Compare actual preview lengths with the pinned legacy loader and sampler.

This reports an implementation difference, not a v2 paper criterion and not
an instruction to apply PHC filtering to GRAB or any other data source.
"""
import ast
import collections
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def legacy_policy(source,dataset_name):
    tree=ast.parse(source)
    cls=next((x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Text2MotionDatasetV2'),None)
    if cls is None:raise ValueError('legacy dataset class missing')
    init=next((x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__'),None)
    if init is None:raise ValueError('legacy dataset initializer missing')
    assignments=[x for x in ast.walk(init) if isinstance(x,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='min_motion_len' for t in x.targets)]
    if len(assignments)!=1 or not isinstance(assignments[0].value,ast.IfExp):raise ValueError('legacy minimum not recognized')
    expr=assignments[0].value
    if ast.unparse(expr.test)!="self.opt.dataset_name == 't2m'":raise ValueError('legacy dataset branch changed')
    if not isinstance(expr.body,ast.Constant) or not isinstance(expr.orelse,ast.Constant):raise ValueError('legacy minimum not literal')
    minimum=expr.body.value if dataset_name=='t2m' else expr.orelse.value
    maxima=[];has_lower=False
    for node in ast.walk(init):
        if not isinstance(node,ast.Compare) or len(node.ops)!=1 or len(node.comparators)!=1:continue
        if ast.unparse(node.left)!='len(motion)':continue
        right=node.comparators[0]
        if isinstance(node.ops[0],ast.Lt) and isinstance(right,ast.Name) and right.id=='min_motion_len':has_lower=True
        if isinstance(node.ops[0],ast.GtE) and isinstance(right,ast.Constant) and isinstance(right.value,int):maxima.append(right.value)
    if not has_lower or len(maxima)!=1 or not isinstance(minimum,int) or not 0<minimum<maxima[0]:
        raise ValueError('legacy whole-motion bounds not recognized')
    return minimum,maxima[0]

def audit_lengths(records,*,root,batch_size,seed,legacy_dataset,legacy_dataset_sha256,legacy_options,legacy_options_sha256):
    from length_bucket_sampler import ExactLengthBatchSampler
    from robot_motion_dataset import RobotMotionDataset
    root=Path(root).resolve(strict=True);legacy=Path(legacy_dataset).resolve(strict=True);opts=Path(legacy_options).resolve(strict=True)
    if digest(legacy)!=legacy_dataset_sha256 or digest(opts)!=legacy_options_sha256:raise ValueError('legacy code/options hash mismatch')
    names=[line.split(':',1)[1].strip() for line in opts.read_text().splitlines() if line.startswith('dataset_name:')]
    if names!=['t2m']:raise ValueError('legacy options do not specify exactly t2m')
    minimum,maximum=legacy_policy(legacy.read_text(),names[0])
    sig=inspect.signature(RobotMotionDataset.__init__)
    custom_min=sig.parameters['min_frames'].default;custom_max=sig.parameters['max_frames'].default
    rows=[];fake_records=[]
    for record in records:
        rel=Path(record['feature_path']);p=(root/rel).resolve(strict=True)
        if rel.is_absolute() or not p.is_relative_to(root) or digest(p)!=record['feature_sha256']:raise ValueError('feature hash or path mismatch')
        with np.load(p,allow_pickle=False) as f:x=f['features'];length=len(x)
        if x.ndim!=2 or x.shape[1]!=263:raise ValueError('invalid features')
        rows.append({'sample_id':record['sample_id'],'split':record['split'],'feature_rows':length,
                     'legacy_t2m_whole_motion_length_eligible':minimum<=length<maximum,
                     'experimental_loader_length_eligible':custom_min<=length<=custom_max})
        if record['split']=='train':
            st=p.stat();fake_records.append({'sample_id':record['sample_id'],'_path':p,'_file_size':st.st_size,'_file_mtime_ns':st.st_mtime_ns})
    dataset=SimpleNamespace(records=fake_records,min_frames=custom_min,max_frames=custom_max)
    sampler=ExactLengthBatchSampler(dataset,batch_size,seed=seed,shuffle=True,drop_last=False)
    batches=[[fake_records[i]['sample_id'] for i in batch] for batch in sampler]
    if digest(legacy)!=legacy_dataset_sha256 or digest(opts)!=legacy_options_sha256:raise RuntimeError('legacy policy source changed')
    return {'legacy_dataset_sha256':legacy_dataset_sha256,'legacy_options_sha256':legacy_options_sha256,
            'legacy_dataset_name':names[0],'legacy_min_inclusive':minimum,'legacy_max_exclusive':maximum,
            'experimental_min_inclusive':custom_min,'experimental_max_inclusive':custom_max,'cases':rows,
            'configured_batch_size':batch_size,'actual_train_batches':batches,
            'actual_batch_size_histogram':dict(collections.Counter(str(len(x)) for x in batches)),
            'training_ready':False,
            'limitations':['Legacy HumanML dataset preprocessing is not a documented v2 filter.',
                           'GRAB is explicitly exempt from v2 feasibility filtering; no exclusion is made here.',
                           'Batch grouping and declared FPS do not establish physical, semantic or source quality.']}
