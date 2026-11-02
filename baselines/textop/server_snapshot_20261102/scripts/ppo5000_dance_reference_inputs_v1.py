"""Authenticate private original VAL dance metadata before a future FK job.

Does not read motion arrays, load a model, launch SDK, or start an evaluation.
Physical outputs and actor acceptance require independent subsequent jobs.
"""
import json
from step4_train_only_protocol import ROOT, owned, read, sha, scope
from filtered15000_vae_completion import closed_job
from ldm100k_scope_contract import RUN, validate_scope_receipt
from ppo5000_dance_val_candidate_v1 import candidates
from ppo5000_dance_reference_plan_v1 import plan


def accepted_metadata(binding):
    if not isinstance(binding,dict) or binding.get('run')!=RUN:
        raise ValueError('incorrect closed VAL scope binding')
    sp=owned(ROOT/'configs/ldm100k-heldout-scope-spec.json')
    assert sha(sp)==binding['spec_sha256']
    spec=read(sp)
    rp=owned(ROOT/'ldm100k_user_handoff_20261028'/RUN/'receipt.json')
    assert sha(rp)==binding['receipt_sha256']
    rec=read(rp);validate_scope_receipt(rec,spec,sha(sp))
    proof=closed_job('ldm100k_heldout_scope_'+RUN,rp,sha(rp),spec['supervisor_sha256'])
    assert proof==binding['closure'] and proof['exit_sha256']==binding['exit_sha256']
    def artifact(name):
        item=rec['artifacts'][name];path=owned(item['path'])
        assert path.parent==rp.parent and path.name==name and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        return read(path)
    source_index=artifact('source_index.private.json');validation=artifact('validation_scope.private.json')
    clips,train,_,train_binding=scope()
    assert len(source_index)==1774 and not ({x['canonical_source'] for x in source_index}&set(train))
    choices=candidates(validation['segments'],source_index,train)
    estimate=plan(choices,train);assert estimate['candidate_count']==len(choices) and choices
    selected={x['canonical_source'] for x in choices};idx={x['canonical_source']:x for x in source_index}
    names={}
    for filename,digest in rec['prepared_manifest_sha256'].items():
        path=owned(filename);assert sha(path)==digest
        with path.open('r',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);k=row['canonical_source']
                if k not in selected:continue
                assert k not in names and row['split']=='val' and row['dof']==29 and row['output_sha256']==idx[k]['output_sha256']
                order=row['joint_names'];assert len(order)==len(set(order))==29
                assert row['frames50']==idx[k]['frames50'] and row['output_file']==idx[k]['output_file']
                names[k]=order
    assert set(names)==selected
    return dict(candidates=choices,source_index=idx,joint_names=names,estimate=estimate,
        train_binding=train_binding,validation_receipt_sha256=sha(rp),validation_closure=proof,
        scope_spec_sha256=sha(sp),validation_scope_sha256=rec['artifacts']['validation_scope.private.json']['sha256'],
        source_index_sha256=rec['artifacts']['source_index.private.json']['sha256'],
        source_arrays_loaded=False,actual_FK=False,physical_result=False)
