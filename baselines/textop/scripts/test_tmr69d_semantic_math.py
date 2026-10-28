"""Independent analytic/numeric cases; never reads real latents or models."""
import hashlib
import json
import sys
import numpy as np
from tmr69d_semantic_math import candidate_blocks,euclidean,retrieval32,frechet_raw,matrix

ids=[hashlib.sha256(('isolated-segment-'+str(i)).encode()).hexdigest() for i in range(188)]
blocks=candidate_blocks(ids)
assert len(blocks)==6 and [len(x['queries']) for x in blocks]==[32]*5+[28]
assert blocks[-1]['reused_distractor_count']==4
assert all(len(x['candidates'])==len(set(x['candidates']))==32 for x in blocks)
assert sorted(i for x in blocks for i in x['queries'])==list(range(188))
reverse=list(reversed(ids));other=candidate_blocks(reverse)
assert [[ids[i] for i in x['candidates']] for x in blocks]==[[reverse[i] for i in x['candidates']] for x in other]
assert candidate_blocks(ids)==blocks
x=np.array([[0.,0.],[3.,4.]],dtype=np.float64)
assert np.array_equal(euclidean(x,x),[[0.,5.],[5.,0.]])
assert np.array_equal(euclidean(x+1e9,x+1e9),[[0.,5.],[5.,0.]])
# A unique paired target is exactly nearest for every one of the188 queries.
grid=np.arange(188,dtype=np.float64)[:,None]
perfect=retrieval32(grid,grid,ids)
assert perfect['Rprecision_top1']==perfect['Rprecision_top3']==1 and perfect['MMdist']==0
assert perfect['actual_queries']==188 and perfect['last_block_scored_queries']==28
# Independent brute-force nearest-neighbour and MM distance oracle.
gen=grid+np.sin(grid)*.2
got=retrieval32(grid,gen,ids)
ranks=[]
for block in blocks:
 for query in block['queries']:
  d=[float(abs(grid[query,0]-gen[candidate,0])) for candidate in block['candidates']]
  target=float(abs(grid[query,0]-gen[query,0]))
  ranks.append(sum(z<target for z in d)+(sum(z==target for z in d)-1)/2)
assert got['Rprecision_top1']==sum(z<1 for z in ranks)/188
assert np.isclose(got['MMdist'],np.abs(grid-gen).mean())
ties=retrieval32(np.zeros((188,2)),np.zeros((188,2)),ids)
assert ties['Rprecision_top1']==ties['Rprecision_top2']==ties['Rprecision_top3']==0
assert all(r==15.5 for b in ties['blocks'] for r in b['zero_based_average_tied_ranks'])
# Exact analytic FID for shifted means and scaled sample covariances.
a=np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])
same=frechet_raw(a,a)
shift=frechet_raw(a,a+np.array([2.,3.]))
scale=frechet_raw(a,2*a)
assert abs(same['FID'])<1e-12 and np.isclose(shift['FID'],13.,atol=1e-12)
assert np.isclose(scale['FID'],np.trace(np.cov(a,rowvar=False,ddof=1)),atol=1e-12)
assert np.isclose(frechet_raw(2*a,a)['FID'],scale['FID'])
degenerate=np.repeat([[1.,2.,3.]],188,axis=0)
assert frechet_raw(degenerate,degenerate)['FID']==0 and frechet_raw(degenerate,degenerate)['covariance_singular']
rank1=np.c_[grid,grid*2,grid*3]
assert frechet_raw(rank1,rank1)['FID']<1e-2 and frechet_raw(rank1,rank1)['covariance_singular']
negative=[]
for label,fn in [
 ('duplicate_segment',lambda:candidate_blocks(ids[:-1]+[ids[0]])),
 ('too_few_candidates',lambda:candidate_blocks(ids[:31])),
 ('nonfinite_input',lambda:matrix(np.array([[1.,np.nan],[2.,3.]]))),
 ('non_float_input',lambda:matrix(np.array([[1,2],[3,4]]))),
 ('not_a_matrix',lambda:matrix(np.zeros(3))),
 ('feature_mismatch',lambda:euclidean(np.zeros((32,2)),np.zeros((32,3)))),
 ('paired_count_mismatch',lambda:retrieval32(np.zeros((188,2)),np.zeros((187,2)),ids)),
 ('fid_population_mismatch',lambda:frechet_raw(np.zeros((188,2)),np.zeros((187,2)))),
 ('fid_singleton',lambda:frechet_raw(np.zeros((1,2)),np.zeros((1,2)))),
]:
 try:fn()
 except (AssertionError,ValueError):negative.append(label)
 else:raise AssertionError('Expected rejection: '+label)
assert 'torch' not in sys.modules
print(json.dumps(dict(status='TMR69D_SEMANTIC_NUMERICAL_KERNEL_FIXTURES_VERIFIED',
 isolated_negative_cases=negative,analytic_fid_cases=5,all188_queries_and_last28_preserved=True,
 reference_schedule_independent_of_input_order=True,actual_robot_latents_read=0,
 actual_models_loaded=0,actual_gpu_contexts=0,actual_trained_semantic_scores=0)))
