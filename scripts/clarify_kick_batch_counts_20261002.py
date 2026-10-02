from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, unquote
import numpy as np
import json, hashlib, shutil, datetime

ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REPORT=ROOT/'实验结果汇报.html'
PROOF=ROOT/'论文复现汇报资料_20261002/生成与验收'
backup=ROOT/'实验结果汇报_Kick单批与历史累计数量澄清前备份.html'
if not backup.exists(): shutil.copy2(REPORT,backup)
s=BeautifulSoup(REPORT.read_text(encoding='utf-8'),'html.parser')
human=ROOT/'人体生成视频'
data=[np.load(human/f'Kick_seed{seed}_original_arrays.npz',allow_pickle=True) for seed in [10,42]]
assert all(d['motion'].shape==(100,22,3,196) for d in data)
hashes=[{hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() for m in d['motion']} for d in data]
overlap=len(hashes[0]&hashes[1]);hints_equal=np.array_equal(data[0]['hint'],data[1]['hint'])
old=s.find(id='kick-batch-count-scope')
if old:old.decompose()
block=BeautifulSoup('''<div class="note" id="kick-batch-count-scope"><b>样本统计口径：Kick单批100条，历史两批累计200条。</b><p>本节论文复现展示的官方Kick原始人体数组含100条，最终G1控制参考库含66条。下方“200条”另指本课题此前两个独立生成批次：seed10有100条、seed42有100条。它不是阶段一单批生成200条，也不是本轮官方参考库重新生成的数量。</p><p>两批虽使用相同Kick文本，随机种子及实际空间提示并不完全相同，不能合并称为一组统一输入条件实验。历史合计194/200的疑似滑脚提醒，应分批报告为97/100和97/100；它不对应上方官方100条样本的质量统计。</p><p><a href="#kick-original-geometric-audit">历史两批数量与几何筛查明细</a>；原始数组、视频和批次来源均保留，不删除或改变已有实验结果。</p></div>''','html.parser').div
s.find(id='generation').find(id='generation-reference-distinction').insert_after(block)
audit=s.find(id='kick-original-geometric-audit')
audit.summary.string='历史归档：两批独立Kick生成，各100条；累计200条及分批几何筛查'
audit.find('p',recursive=False).string='以下是此前独立推理的历史归档：seed10和seed42各100条、每条9.8秒。两批不是同一组空间条件输入，本次未重新生成；也未混入官方66条Kick PPO基线。单批统计以100为分母，200只用于历史两批的累计文件与样本清单。下列194/200等合计提醒不适用于上方作者发布的官方100条Kick。'
hero=s.header.select('.hero-grid>div')[2]
hero.strong.string='18段'
hero.span.string='六类官方人体展示视频；另存历史两批Kick各100段'

for p in s.find(id='dreamcontrol-input-reference').find_all('p',recursive=False):
    if '本次复现实际执行了什么' in p.get_text():
        for text in list(p.find_all(string=True)):
            text.replace_with(str(text).replace('此前本课题自行生成的两批Kick共200条另行归档','此前本课题seed10、seed42分别生成100条Kick，两批历史合计200条另行归档'))

art=s.find(id='artifact-input-reference-note')
art.append(' 每批是独立的100条；历史合计200条不能作为本轮阶段一单批样本量。')
REPORT.write_text(str(s),encoding='utf-8')
missing=[]
for tag in s.select('[href],[src],[poster]'):
    for attr in ['href','src','poster']:
        val=tag.get(attr)
        if not val:continue
        u=urlsplit(val)
        if u.scheme:continue
        if u.path and not (ROOT/unquote(u.path)).exists():missing.append(val)
        elif not u.path and u.fragment and not s.find(id=u.fragment):missing.append(val)
assert not missing,missing
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'official_kick_human_samples':100,'official_kick_robot_references':66,'historical_batches':[{'seed':10,'count':100,'array':'Kick_seed10_original_arrays.npz'},{'seed':42,'count':100,'array':'Kick_seed42_original_arrays.npz'}],'historical_archive_total':200,'cross_batch_exact_motion_hash_overlap':overlap,'cross_batch_spatial_hints_identical':bool(hints_equal),'generation_video_count':len(s.find(id='generation').find_all('video')),'robot_video_count':len(s.find(id='videos').find_all('video')),'missing_links':missing,'supersedes_report_hash_receipt':'扩散输入与参考动作说明核验.json'}
(PROOF/'Kick单批与历史累计数量澄清.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__),PROOF/Path(__file__).name)
print(json.dumps(receipt,ensure_ascii=False))
