"""Summarize finished upstream logs without manufacturing experiment scores."""
import argparse
import html
import json
import re
from pathlib import Path

def parse(path):
    if not path.exists():
        return None
    text = path.read_text(errors='replace')
    sections = re.split(r'========== (.*?) Summary ==========\n', text)
    metrics = {}
    for i in range(1, len(sections), 2):
        section, body = sections[i], sections[i+1]
        line = next((v for v in body.splitlines() if v.startswith('---> [vald]')), '')
        if section == 'R_precision':
            for k, mean, ci in re.findall(r'\(top (\d)\) Mean: ([\d.e+-]+) CInt: ([\d.e+-]+)', line):
                metrics['R@'+k] = {'mean':float(mean), 'ci95':float(ci)}
        else:
            match = re.search(r'Mean: ([\d.e+-]+) CInterval: ([\d.e+-]+)', line)
            if match:
                metrics[section] = {'mean':float(match[1]), 'ci95':float(match[2])}
    return {'complete_summary': 'skating' in metrics and 'FID' in metrics,
            'metrics':metrics, 'replications':len(re.findall(r'Replication \d+',text)), 'log':str(path)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    result={'baseline':'CLoSD / ICLR 2025', 'released_eval':parse(root/'released_evaluation/eval.log'),
            'fresh_eval':parse(root/'fresh_generation/eval.log'),
            'pipeline_state':(root/'pipeline.state').read_text().strip() if (root/'pipeline.state').exists() else 'NOT_STARTED'}
    if (root/'fresh_generation/receipt.json').exists():
        result['generation']=json.loads((root/'fresh_generation/receipt.json').read_text())
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    paper={'R@1':.385,'R@2':.578,'R@3':.693,'FID':1.755,'Matching Score':3.65,'Diversity':8.23,
           'penetration':.021,'floating':20.1,'skating':.0012}
    labels={'Matching Score':'文本—动作匹配距离 ↓','FID':'动作分布 FID ↓','Diversity':'动作多样性（接近真实分布）',
            'penetration':'地面穿透（mm）↓','floating':'最低关节离地距离（mm）↓','skating':'地面滑移（mm）↓'}
    def value(run,k):
        metric=((run or {}).get('metrics') or {}).get(k)
        return f"{metric['mean']:.4f} ± {metric['ci95']:.4f}" if metric else '尚无完成结果'
    rows=''.join('<tr><td>'+html.escape(labels.get(k,k+' ↑'))+'</td><td>'+str(v)+'</td><td>'+value(result['released_eval'],k)+'</td><td>'+value(result['fresh_eval'],k)+'</td></tr>' for k,v in paper.items())
    receipt=result.get('generation')
    generation=html.escape(json.dumps(receipt,indent=2,ensure_ascii=False)) if receipt else '尚未完成新动作采集。'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CLoSD 基线复现</title>
<style>body{font:16px/1.7 Arial,"Microsoft YaHei",sans-serif;max-width:1200px;margin:35px auto;padding:0 25px;color:#183044}h1,h2{color:#123d5b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd7df;padding:9px;text-align:left}th{background:#eaf1f6}aside{background:#fff5de;border-left:4px solid #dfa532;padding:15px}pre{white-space:pre-wrap;background:#f2f5f7;padding:15px}a{color:#086aaf}video{max-width:100%}@media print{body{margin:0;font-size:11pt}details{display:block}table{font-size:9pt}}</style>
<h1>CLoSD：条件动作生成与物理执行基线复现</h1><p>日期：2026-10-10 · 研究方向：面向人形机器人的物理引导条件动作生成</p>
<p><a href="https://guytevet.github.io/CLoSD-page/">论文项目（ICLR 2025 Spotlight）</a> · <a href="https://github.com/GuyTevet/CLoSD">官方代码</a></p>
<h2>为什么选它</h2><p>CLoSD 的 DiP 扩散规划器根据文本和条件生成短期参考动作；物理跟踪器执行这些动作，再将实际状态反馈给下一轮生成。其公开发布包含生成器、控制器、数据与评价程序，适合在现有环境建立可运行的生成—物理执行基线。此次重点复现 HumanML3D 文本动作分支；已有多任务控制实验单独保留。</p>
<aside>本次使用作者发布的检查点进行推理与评测，不是从零重训整篇论文。执行对象是 SMPL 仿真人体角色，不是真实 G1。本次未做 sim-to-real，也不能据此宣称生成数据提高了机器人训练效果。</aside>
<h2>实验状态</h2><p>STATE</p><p>源码固定 d8950bdb78312fcd2844f336b3fc079e7368c87f；Isaac Gym Preview4；GPU0 RTX A6000；新采集配置 4096 并行环境、20,000 条保留动作、seed10。复用原有环境与资产，不下载重复大型数据。</p>
<h2>论文表3与本地结果</h2><p>第二列是论文报告值；第三列重新评价作者发布的动作；第四列只在服务器新采集与评价全部完成后填写。± 表示官方十次评测重采样的95%区间，不是十次独立训练。</p>
<table><thead><tr><th>指标</th><th>论文表3</th><th>本地重评公开动作</th><th>本地新生成动作</th></tr></thead><tbody>ROWS</tbody></table>
<h2>必须保留的边界</h2><ul><li>文本分支以测试动作前缀初始化；不是从任意零姿态仅凭文本生成。</li><li>官方采集只保留完整、未提前终止的回合。动作质量分数属于筛选后的动作集，不能作为所有尝试的成功率。另记 reset/termination 事件；两者不是经过去重的按文本成功率。</li><li>物理指标基于关节坐标与地面阈值，不能证明关节力矩、物体接触力或硬件安全。跳跃中的合理腾空也可能增加 floating 指标。</li><li>多样性数值不是越大越好；R-Precision/FID 使用固定评估编码器，不等价于人工语义评价。</li><li>官方 mean_penetration 辅助项实际包含求和，不作为跨样本规模可比的平均穿透指标。本表只用论文表3的 penetration。</li><li>本次无新训练，不作新增数据训练收益、泛化提升或全部复杂技能覆盖的结论。</li></ul>
<h2>新采集原始验收记录</h2><pre>GENERATION</pre>
<h2>后续对比怎么用</h2><p>固定 HumanML3D 分割、文本与初始前缀、生成随机种子、控制器检查点和采样预算；分别报告生成质量、物理指标、未筛选执行失败、合格比例与耗时。在这些条件相同的基础上，再评价提出的生成或物理修正模块。不要只比较筛选后漂亮的动作。</p><p>本次忠实运行论文的通用 HumanML3D 测试基准，其中包含日常动作；这不等于将你的研究任务改成行走等简单动作，也不证明舞蹈、体育或体操子任务均已充分覆盖。复杂技能的专门对比应另建预先固定的测试子集，明确文本筛选和动作审核标准。</p>
<p>本地源代码与配置：scripts/closd_baseline_20261010.sh、scripts/closd_t2m_record_20261010.py、docs/closd_baseline_20261010.md。原始动作受各数据集许可约束，仅保留在服务器，不上传公共仓库。</p></html>'''
    preview_manifest=root/'previews/manifest.json'
    preview_dir='previews'
    if not preview_manifest.exists():
        preview_manifest=root/'smoke_previews/manifest.json'
        preview_dir='smoke_previews'
    if preview_manifest.exists():
        clips=json.loads(preview_manifest.read_text())
        block='<h2>新生成动作预览</h2><p>以下固定展示数组前3条；是已执行轨迹转换后的骨架回放，不是实机视频或实时仿真截图。没有按效果挑选。</p>'
        if preview_dir=='smoke_previews':
            block+='<aside>当前展示4环境小规模核验的前3条动作，不是完整20,000条采集的结果。</aside>'
        for clip in clips:
            block+='<p>'+html.escape(clip['caption'])+'</p><video controls preload="metadata" width="600" src="'+preview_dir+'/'+clip['file']+'"></video>'
        page=page.replace('<h2>必须保留的边界</h2>',block+'<h2>必须保留的边界</h2>')
    page=page.replace('STATE',html.escape(result['pipeline_state'])).replace('ROWS',rows).replace('GENERATION',generation)
    (root/'baseline_report.html').write_text(page,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
