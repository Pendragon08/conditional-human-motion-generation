from pathlib import Path
import ast,shutil
T=Path(__file__).resolve().parent;R=T.parent/'phygile_reproduction_20261017/conditional-human-motion-generation';B=R/'baselines/textop';M=B/'sources/filtered15000_handoff_maintenance_20261027'
for name in ['finalize_filtered15000_ldm_preparation.py','prepare_ldm_preparation_sync.py']:
 dst=M/name;assert not dst.exists();shutil.copy2(T/name,dst)
s=(T/'sync_filtered15000_handoff_tools.py').read_text(encoding='utf-8')
s=s.replace("E=T/'filtered15000_handoff_tools_sync_v1'","E=T/'filtered15000_ldm_preparation_sync_v1'")
names=['.gitattributes','README.md','CONTINUATION.md','FILTERED15000_HANDOFF_ENVIRONMENT.md',
 'provider_tools/activate_filtered15000_corpus.py','scripts/filtered15000_ldm_runtime.py','scripts/full_ldm_history.py',
 'configs/filtered15000-ldm-plan.json','configs/teacher15000-full-to-generator-autostart-20261027.json',
 'reports/filtered15000_ldm_preparation_20261027.md','reports/teacher15000_direct_strict_20261027.md',
 'results/filtered15000_handoff_tools_20261027/ldm_preparation_cpu_v1.json','results/filtered15000_handoff_tools_20261027/ldm_runtime_deploy_v1.json',
 'results/teacher15000_direct_strict_20261027/current.json','results/teacher15000_direct_strict_20261027/teacher15000_direct_full_v1/progress.json']
start=s.index('names=[');end=s.index('names+=',start);s=s[:start]+'names='+repr(names)+'\n'+s[end:]
s=s.replace('Stage disabled closed-result audit and actual-pass generator handoff entries','Implement actual-pass corpus activation and stage full300k DDPM history runtime')
s=s.replace('results/filtered15000_handoff_tools_20261027/deploy_v1.json','results/filtered15000_handoff_tools_20261027/ldm_runtime_deploy_v1.json')
s=s.replace('GitHub同步记录_严格筛查后生成器入口准备_20261027.json','GitHub同步记录_完整LDM数值入口与语料激活桥_20261027.json')
ast.parse(s);p=T/'sync_filtered15000_ldm_preparation.py';assert not p.exists();p.write_text(s,encoding='utf-8')
print('New explicit incremental sync source prepared')
