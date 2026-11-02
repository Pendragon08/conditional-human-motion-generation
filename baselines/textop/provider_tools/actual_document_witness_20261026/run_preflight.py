import pathlib,sys,json,traceback,time
W=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(B/'provider_tools'))
from session import remote
try:
    raw=remote((W/'preflight_remote.py').read_text(encoding='utf-8'),timeout=180)
    (W/'preflight_raw.json').write_text(raw,encoding='utf-8')
    result=json.loads(raw)
    assert result['status']=='PREFLIGHT_PASSED'
    for rel,h in result['source_sha256'].items():
        import hashlib
        assert hashlib.sha256((B/rel).read_bytes()).hexdigest()==h,rel
    result['local_and_remote_bound_sources_match']=True
    (W/'preflight.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'closed_trials':result['author_private_verified']['closed_trials'],'natural':result['author_private_verified']['natural_complete'],'failed':result['author_private_verified']['failed'],'actual_control_calls':result['author']['receipt']['actual_vectorized_control_calls'],'actual_scene_physics_steps':result['author']['receipt']['actual_scene_physics_steps'],'author_exit':result['author']['exit']['child_exit_code'],'lock':result['common_lock_available'],'gpu1_free_mib':result['resources']['free_gpu1_mib'],'source_count':len(result['source_sha256'])}))
except BaseException:
    (W/'preflight_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
