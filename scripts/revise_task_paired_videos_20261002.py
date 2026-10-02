from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from bs4 import BeautifulSoup
import json, hashlib, shutil, os, copy

ROOT = Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REPO = Path(r'D:\具身智能\conditional-human-motion-generation')
REPORT = ROOT / '实验结果汇报.html'
PACK = ROOT / '论文复现汇报资料_20261002'
PROOF = PACK / '生成与验收'
PROOF.mkdir(exist_ok=True)
BACKUP = ROOT / '实验结果汇报_按任务配对视频修订前备份.html'
if not BACKUP.exists(): shutil.copy2(REPORT, BACKUP)
soup = BeautifulSoup(REPORT.read_text(encoding='utf-8'), 'html.parser')
def node(html): return BeautifulSoup(html, 'html.parser')
def link(path): return quote(path.relative_to(ROOT).as_posix(), safe='/')
def save_linked(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try: os.link(src, dest)
        except OSError: shutil.copy2(src, dest)
    assert hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dest.read_bytes()).digest()

ground_src = REPO / 'media/r3_official_ground_pick_8192_eval.mp4'
ground_pack = PACK / 'media/r3_official_ground_pick_8192_eval.mp4'
ground_local = ROOT / '机器人执行视频/01_Isaac_Lab_官方基线/r3_official_ground_pick_8192_eval.mp4'
save_linked(ground_src, ground_pack); save_linked(ground_src, ground_local)
for name in ['r3_dreamcontrol_official_Ground-Pick_8192_eval.json', 'r3_dreamcontrol_Ground-Pick_eval_failure_20261002.json']:
    shutil.copy2(REPO / 'results' / name, PACK / 'results' / name)
shutil.copy2(REPO / 'reports/03_dreamcontrol_multitask_baselines.md', PACK / 'reports/03_dreamcontrol_multitask_baselines.md')

tasks = ['Kick', 'Punch', 'Jump', 'Sit', 'Pick', 'Bimanual-Pick', 'Ground-Pick']
scores = [85.3, 87.2, 100, 100, 93.9, 99.9, 38.1]
conclusions = {
 'Kick': ('部分迁移，稳定性仍不足', '48/66条到达虚拟目标且完整10秒不摔倒（72.7%）；9条摔倒。此处未要求实体打击接触。', 'partial'),
 'Punch': ('部分迁移，通过比例偏低', '33/100条到达虚拟目标且完整10秒不摔倒（33.0%）；26条摔倒。此处未要求实体打击接触。', 'partial'),
 'Jump': ('本次迁移未通过', '唯一参考约2.9秒摔倒，未完成再次腾空与稳定落地。', 'failed'),
 'Sit': ('站稳，但坐下接触未通过', '唯一参考运行10秒并达位置目标；持续座椅接触0/1。训练座椅与参考目标不对齐，保留该场景差异。', 'failed'),
 'Pick': ('站稳，但取物未通过', '67/67条完整运行10秒；右手接触并相对抬物≥5cm持续1秒为0/67。', 'failed'),
 'Bimanual-Pick': ('站稳，但双手抬物未通过', '唯一参考完整运行10秒；双手接触并相对抬物≥5cm持续1秒为0/1。', 'failed'),
 'Ground-Pick': ('MuJoCo尚未验证', 'Isaac Lab训练与官方评估刚完成；尚无跨仿真成绩或视频。', 'pending'),
}

videos = soup.find(id='videos')
old_cards = list(videos.select('.video-card'))
base_cards = {c.h3.get_text().split('｜')[0]: c.extract() for c in old_cards if c.get('data-group') == 'baseline'}
sim_cards = {t: [] for t in tasks}
for c in old_cards:
    if c.get('data-group') != 'sim': continue
    title = c.h3.get_text()
    for t in sorted(tasks, key=len, reverse=True):
        if f'{t}_reference_' in title:
            sim_cards[t].append(c.extract()); break
robust_cards = [c.extract() for c in old_cards if c.get('data-group') == 'robust']
base_cards['Ground-Pick'] = node(f'<article class="video-card"><h3>Isaac Lab｜官方评估</h3><video controls preload="none"><source src="{link(ground_pack)}" type="video/mp4"></video><p>38.1%；1000环境、500帧；官方USD I/O适配。501帧，10.02秒。</p><a href="{link(ground_local)}">本地视频文件</a></article>').article
videos.clear()
videos.append(node('<div class="eyebrow">按任务配对 · 28段执行视频</div><h2>Isaac Lab执行与MuJoCo迁移：逐任务展示</h2><p>每个已完成控制任务均单独列出：左侧为Isaac Lab官方评估，右侧为冻结同一策略后的MuJoCo执行。视频用于展示动作；迁移结论依据完整试验统计，不依据某一段展示视频。两平台的试验口径不同，不直接相减成功率。</p>').contents[0])
# Append the remaining introductory elements separately.
intro = node('<h2>Isaac Lab执行与MuJoCo迁移：逐任务展示</h2><p>每个已完成控制任务均单独列出：左侧为Isaac Lab官方评估，右侧为冻结同一策略后的MuJoCo执行。视频用于展示动作；迁移结论依据完整试验统计，不依据某一段展示视频。两平台的试验口径不同，不直接相减成功率。</p>')
for el in list(intro.contents): videos.append(el)
nav = soup.new_tag('div', attrs={'class': 'task-links'})
for t in tasks:
    a = soup.new_tag('a', href=f'#task-video-{t}'); a.string = t; nav.append(a)
videos.append(nav)
for t, score in zip(tasks, scores):
    box = soup.new_tag('div', attrs={'class': 'task-video-block', 'id': f'task-video-{t}', 'data-task': t})
    heading = soup.new_tag('h3'); heading.string = t; box.append(heading)
    label, explanation, status = conclusions[t]
    box.append(node(f'<p class="transfer-result {status}"><b>迁移结论：{label}</b><br>{explanation}</p>').p)
    grid = soup.new_tag('div', attrs={'class': 'video-grid'})
    c = base_cards[t]; c.attrs.pop('data-group', None); c.h3.string = f'Isaac Lab｜{t}官方评估'
    if t in ['Sit', 'Bimanual-Pick']:
        c.p.string = f'{score:.1f}%；1000环境、500帧；官方--disable_fabric USD I/O适配，非默认Fabric路径。'
    grid.append(c)
    if sim_cards[t]:
        primary = sim_cards[t][0]; primary.attrs.pop('data-group', None)
        primary.h3.string = f'MuJoCo｜{t}参考000'
        primary.p.string = '固定参考000的冻结策略真实物理执行；失败视频保留。完整统计见迁移效果表。'
        grid.append(primary)
    else:
        grid.append(node('<article class="video-card pending-video"><h3>MuJoCo｜待验证</h3><div class="pending-placeholder">尚无跨仿真视频</div><p>未运行该任务的MuJoCo验证；不展示其他任务或Isaac画面作为替代。</p></article>').article)
    box.append(grid)
    if len(sim_cards[t]) > 1:
        detail = soup.new_tag('details'); summary = soup.new_tag('summary'); summary.string = f'{t}补充MuJoCo视频：固定参考001'; detail.append(summary)
        for extra in sim_cards[t][1:]:
            extra.attrs.pop('data-group', None); extra.h3.string = f'MuJoCo｜{t}参考001'; detail.append(extra)
        box.append(detail)
    videos.append(box)
detail = soup.new_tag('details', id='robust-video-supplement')
summary = soup.new_tag('summary'); summary.string = 'Kick / Punch：12段扰动与实体接触补充视频'; detail.append(summary)
grid = soup.new_tag('div', attrs={'class':'video-grid'})
for c in robust_cards: c.attrs.pop('data-group', None); grid.append(c)
detail.append(grid); videos.append(detail)
videos.append(node('<p class="source">所有28段机器人视频均已保存在本地“机器人执行视频”文件夹。CLoSD两组官方权重评估未录制视频，不用其他画面替代。尚未完成的Ground-Pick-Top、Button-Press、Open-Drawer、Open-Door没有正式结果视频。</p>').p)

sim = soup.find(id='sim')
# Keep the paper rationale, protocol, raw evidence and interface audit.
sim.h2.string = 'MuJoCo迁移效果：任务完成与稳定性分别验收'
raw_table = sim.find('div', class_='tablewrap', recursive=False).extract()
raw_note = sim.find('div', class_='note', recursive=False).extract()
raw_detail = soup.new_tag('details', id='sim-raw-statistics')
summary = soup.new_tag('summary'); summary.string = '展开原始统计：目标达标、10秒稳定、接触检查和摔倒数'; raw_detail.append(summary); raw_detail.append(raw_table); raw_detail.append(raw_note)
purpose = sim.find(id='sim-purpose-paper')
table_html = '<div id="sim-effect-summary"><h3>迁移效果结论</h3><div class="note"><b>判定依据：任务是否真正完成，且执行过程是否稳定。</b>机器人站立10秒或达到参考位置，不能单独说明坐下、抓取或打击接触成功。当前六项测试均未建立可靠迁移结论；Ground-Pick尚待跨仿真验证。</div><div class="tablewrap"><table><thead><tr><th>任务</th><th>MuJoCo关键结果</th><th>迁移结论 / 视频</th></tr></thead><tbody>'
for t in tasks:
    label, explanation, status = conclusions[t]
    badge = 'pending' if status == 'pending' else ('running' if status == 'partial' else 'fail')
    table_html += f'<tr><td><b>{t}</b></td><td>{explanation}</td><td><span class="badge {badge}">{label}</span><br><a href="#task-video-{t}">Isaac Lab / MuJoCo并列视频</a></td></tr>'
table_html += '</tbody></table></div><p class="source">Kick/Punch统计虚拟目标到达＋完整10秒不摔倒，实体接触成绩另见“扰动与实体接触”。Jump/Sit/Bimanual各只有一条参考，不能据此估计未见动作泛化率。MuJoCo独立10秒试验与Isaac 1000环境自动重置评分不可直接比较。</p></div>'
effect = node(table_html).div; purpose.insert_after(effect); effect.insert_after(raw_detail)

# Update only the baseline snapshot; retain six-task human generation and exported PPO curves.
baseline = soup.find(id='baseline'); baseline.h2.string = 'DreamControl：七项官方控制基线'
baseline.find('p', recursive=False).string = '七项均采用官方原参考库、8192个训练环境、2000次PPO迭代、seed42；完成至第1999轮后，用官方1000环境、500帧评估。Punch曾中断，从第1600轮恢复。'
table = baseline.find('table')
for row in table.tbody.find_all('tr'):
    task = row.td.get_text(strip=True); a=row.find('a')
    if a: a['href']=f'#task-video-{task}'; a.attrs.pop('data-select',None)
table.tbody.append(node('<tr><td>Ground-Pick</td><td>1</td><td><span class="metric">38.1%</span></td><td>USD显示读写适配</td><td>完成</td><td><a href="#task-video-Ground-Pick">查看视频</a></td></tr>').tr)
pending_table = baseline.find_all('table')[1]
for row in list(pending_table.tbody.find_all('tr')):
    if row.td.get_text(strip=True) == 'Ground-Pick': row.decompose()
    elif row.td.get_text(strip=True) == 'Ground-Pick-Top':
        cells = row.find_all('td'); cells[1].string='训练中'; cells[2].string='已由原唯一队列接续；2026-10-02 17:54核对第18/2000轮'
for p in baseline.find_all('p', recursive=False):
    if 'Sit和Bimanual默认视频' in p.get_text():
        p.append(' Ground-Pick也在默认视频初始化native错误后采用同一官方USD I/O适配；原失败日志保留，最终评估正常退出，模型未重训。具体native错误根因尚未证实。')
header = soup.header
header.select('.hero-grid strong')[0].string = '7 / 11'
header.find('p',recursive=False).clear()
header.find('p',recursive=False).append('人体动作生成方法、论文控制基线与跨仿真验证')
for p in header.find_all('p',recursive=False):
    if '实验成绩截至' in p.get_text(): p.clear(); p.append('实验成绩截至2026年10月2日17:54（北京时间）；逐任务视频展示已更新。文件夹沿用用户指定的20260929名称。')
overview = soup.find(id='overview')
for text in list(overview.find_all(string=True)):
    new=str(text).replace('六项官方控制任务已有结果','七项官方控制任务已有结果').replace('6项训练＋官方评估＋视频','7项训练＋官方评估＋视频').replace('OmniControl官方预训练模型的两批Kick输出；固定前3条，共6段','六类官方原始人体样本342条，固定样本000–002共18段展示；另保存自生成Kick 200段')
    if new != str(text): text.replace_with(new)
style = soup.new_tag('style'); style.string = '.task-links{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}.task-links a{padding:7px 13px;border-radius:20px;background:#e7f1f6}.task-video-block{padding:24px 0;border-top:2px solid #e3ebf0;scroll-margin-top:18px}.task-video-block>h3{font-size:24px}.transfer-result{padding:14px 18px;border-left:4px solid #ae7047;background:#fff5ed;border-radius:8px;font-size:14px}.transfer-result.partial{border-color:#b98e36;background:#fff8e8}.transfer-result.pending{border-color:#728397;background:#eef2f6}.pending-placeholder{aspect-ratio:16/9;display:grid;place-items:center;background:#e9eef3;color:#687b8c;border-radius:8px;font-size:20px}.task-video-block details .video-card{max-width:680px}.task-video-block .video-card>p{font-size:13px}#sim-effect-summary td:nth-child(1){width:12%}#sim-effect-summary td:nth-child(2){width:54%}@media(max-width:700px){.task-video-block{padding:14px 0}#sim-effect-summary table{min-width:660px}}'
soup.head.append(style)
REPORT.write_text(str(soup),encoding='utf-8')

manifest_path = ROOT / '机器人执行视频/视频清单.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
if not any(v['file'].endswith(ground_local.name) for v in manifest['videos']):
    manifest['videos'].append({'file':ground_local.relative_to(ROOT/'机器人执行视频').as_posix(),'category':'01_Isaac_Lab_官方基线','duration_seconds':10.02,'sha256':hashlib.sha256(ground_local.read_bytes()).hexdigest(),'source':ground_pack.relative_to(ROOT).as_posix()})
manifest['total_videos']=len(manifest['videos']); manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# Independent offline index now lands on the task-paired report.
index=ROOT/'机器人执行视频/按任务配对观看.html'
index.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>机器人执行与MuJoCo迁移</title><h1>按任务配对观看</h1>'+''.join(f'<p><a href="../{quote(REPORT.name)}#task-video-{t}">{t}：Isaac Lab执行 / MuJoCo迁移</a></p>' for t in tasks)+'</html>',encoding='utf-8')
missing=[]
for tag in soup.select('[href],[src]'):
    for attr in ['href','src']:
        value=tag.get(attr)
        if not value or value.startswith(('#','http:','https:','data:','javascript:')):continue
        rel=unquote(urlsplit(value).path)
        if not (ROOT/rel).exists():missing.append(rel)
assert not missing,missing
receipt={'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'paired_tasks':tasks,'isaac_videos':7,'mujoco_primary_videos':6,'mujoco_extra_videos':3,'robust_contact_videos':12,'ground_pick_mujoco_status':'not_tested','human_inline_videos':len(soup.find(id='generation').find_all('video')),'total_inline_videos':len(soup.find_all('video')),'missing_links':missing,'source_repo_commit':'61efa6d'}
(PROOF/'逐任务配对展示修订.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
