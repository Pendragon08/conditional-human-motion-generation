import sys,pathlib,json,subprocess,hashlib
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import REPO,remote
BASE=REPO/'baselines/textop';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
spec={'base':'warm reuse gmr_amass_20261021; no pip changes','python':'/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python','required_versions':json.loads((REPO/'baselines/phygile/configs/gmr-amass-env-spec.json').read_text())['required_versions'],'env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},'gmr_root':'/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR','body_models':'/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead','paper_configuration':{'dof':29,'fps_retarget':30,'fps_output':50,'contact_height':0.2,'contact_squared_displacement':0.002,'feature_dim':69,'physical_filter_required':True},'validation':'seeded CPU SMPL-X forward and one GMR IK frame, then raw motion replay; not dynamic filtering'}
sp=BASE/'configs/data-prep-env-spec.json';sp.write_text(json.dumps(spec,indent=2))
sh=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
with (BASE/'SERVER_NOTES.md').open('a') as f:f.write(f'\n### env: textop-data-prep@{sh}\nWarm reuse packages from gmr_amass_20261021 (old dependency spec141547bf), no package/environment mutation. New spec records licensed body-model availability and paper69D target. Source and model hash witness pending. Resources CPU4, no CUDA. MVAE57D is SIGSTOP-held pending data preparation.\n')
subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(sp),'ouyanghao@10.147.17.32:'+ROOT+'/configs/data-prep-env-spec.json'],check=True)
print('DATA_PREP_SPEC',sh)
