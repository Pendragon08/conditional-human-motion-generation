"""Explicit diagnostic protocol on trained, raw mean latents; no model imports.

All query segments are scored. A short final block keeps its queries and uses
fixed earlier segments solely as distractors, so every query has 32 candidates.
This is declared adaptation, not an assertion of the author's unknown grouping.
"""
import hashlib


def candidate_blocks(segment_ids):
    assert len(segment_ids) >= 32 and len(set(segment_ids)) == len(segment_ids)
    assert all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in segment_ids)
    order = sorted(range(len(segment_ids)), key=lambda i: (
        hashlib.sha256(('textop-tmr69d-candidates-v1:'+segment_ids[i]).encode()).hexdigest(), segment_ids[i]))
    blocks = []
    for start in range(0,len(order),32):
        queries = order[start:start+32]
        fillers = [i for i in order if i not in queries][:32-len(queries)]
        candidates = queries+fillers
        assert len(candidates)==len(set(candidates))==32
        blocks.append(dict(queries=queries,candidates=candidates,scored_query_count=len(queries),
                           reused_distractor_count=len(fillers)))
    assert sorted(i for row in blocks for i in row['queries']) == list(range(len(segment_ids)))
    return blocks


def matrix(value):
    import numpy as np
    x = np.asarray(value)
    assert x.ndim==2 and x.shape[0]>=2 and x.shape[1]>=1 and x.dtype.kind=='f'
    assert np.isfinite(x).all()
    return np.array(x,dtype=np.float64,order='C',copy=True)


def euclidean(x,y):
    import numpy as np
    x,y=matrix(x),matrix(y)
    assert x.shape[1]==y.shape[1]
    # Direct differences avoid cancellation of large, close raw latents.
    return np.sqrt(np.square(x[:,None,:]-y[None,:,:]).sum(axis=2))


def retrieval32(texts,motions,segment_ids):
    import numpy as np
    t,m=matrix(texts),matrix(motions)
    assert t.shape==m.shape and len(t)==len(segment_ids)
    ranks=[];matches=[];details=[]
    for row in candidate_blocks(segment_ids):
        q,c=row['queries'],row['candidates']
        distances=euclidean(t[q],m[c])
        block_ranks=[]
        for position,index in enumerate(q):
            assert c[position]==index
            target=distances[position,position]
            less=int((distances[position]<target).sum())
            ties=int((distances[position]==target).sum())
            assert ties>=1
            # The upstream retrieval metrics average exactly tied ranks.
            rank=less+(ties-1)/2
            ranks.append(rank);matches.append(float(target));block_ranks.append(rank)
        details.append(dict(scored_queries=len(q),candidate_count=32,
            reused_distractors=row['reused_distractor_count'],zero_based_average_tied_ranks=block_ranks))
    assert len(ranks)==len(t)
    return dict(Rprecision_top1=float(np.mean(np.asarray(ranks)<1)),
        Rprecision_top2=float(np.mean(np.asarray(ranks)<2)),Rprecision_top3=float(np.mean(np.asarray(ranks)<3)),
        MMdist=float(np.mean(matches)),actual_queries=len(t),candidate_count_per_query=32,
        exact_tie_policy='zero_based_average_rank',raw_mean_latents=True,per_sample_L2_normalized=False,
        target_policy='exact_paired_annotation_segment_only',last_block_scored_queries=details[-1]['scored_queries'],
        last_block_reused_distractors=details[-1]['reused_distractors'],blocks=details)


def frechet_raw(reference,generated):
    """Sample covariance(ddof1), PSD square-root trace, float64; supports rank loss."""
    import numpy as np
    a,b=matrix(reference),matrix(generated)
    assert a.shape==b.shape
    mean_a,mean_b=a.mean(0),b.mean(0)
    ca=np.atleast_2d(np.cov(a,rowvar=False,ddof=1))
    cb=np.atleast_2d(np.cov(b,rowvar=False,ddof=1))

    def psd_eigh(value):
        value=(value+value.T)*.5
        eig,vec=np.linalg.eigh(value)
        assert np.isfinite(eig).all()
        tolerance=1e-10*max(1.,float(np.max(np.abs(eig))))
        assert float(eig.min())>=-tolerance, 'Substantial negative covariance eigenvalue'
        return np.maximum(eig,0.),vec,tolerance

    va,ua,ta=psd_eigh(ca)
    vb,_,tb=psd_eigh(cb)
    root=(ua*np.sqrt(va))@ua.T
    middle=root@cb@root
    vm,_,tm=psd_eigh(middle)
    mean_term=float(np.square(mean_a-mean_b).sum())
    covariance_term=float(np.trace(ca)+np.trace(cb)-2*np.sqrt(vm).sum())
    raw=mean_term+covariance_term
    tolerance=1e-7*max(1.,float(np.trace(ca)+np.trace(cb)))
    assert raw>=-tolerance, 'Substantial negative Frechet distance'
    return dict(FID=max(0.,raw),raw_unclamped_value=raw,finite_roundoff_tolerance=tolerance,
        covariance_ddof=1,feature_dimensions=a.shape[1],reference_count=len(a),generated_count=len(b),
        reference_covariance_effective_rank=int((va>ta).sum()),generated_covariance_effective_rank=int((vb>tb).sum()),
        covariance_singular=bool((va<=ta).any() or (vb<=tb).any()),
        psd_negative_roundoff_tolerances=[ta,tb,tm],per_sample_L2_normalized=False)


def paired_seed_metrics(texts,reference,generated,segment_ids):
    assert len(segment_ids)==188, 'This protocol describes the declared188-segment diagnostic'
    retrieval=retrieval32(texts,generated,segment_ids)
    return dict(retrieval=retrieval,frechet=frechet_raw(reference,generated),
        scope='32_original_VAL_families_188_independent_annotation_segments',
        continuous_transition_or_full_validation_claimed=False,
        author_exact_candidate_schedule_claimed=False)
