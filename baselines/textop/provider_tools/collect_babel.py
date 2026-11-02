"""Publish aggregate BABEL provenance, not licensed annotations or pair indexes."""
import datetime,html,json,pathlib,sys
from session import REPO
BASE=REPO/'baselines/textop';DEST=BASE/'results/data_preparation_20261025'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
sys.stdout.reconfigure(encoding='utf-8')

def main():
    acquisition=json.loads((DEST/'babel_acquisition.json').read_text(encoding='utf-8'))
    audit=json.loads((DEST/'babel_alignment.json').read_text(encoding='utf-8'))
    install=json.loads((DEST/'babel_server_install.json').read_text(encoding='utf-8'))
    assert audit['mismatch_sequences']==0 and install['files_verified']==7
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    report=f'''# BABEL 原始标注下载、服务器安装与TextOp配对核查

更新时间：{now}。

**BABEL登录与独立下载已完成。TextOp发布包全部8,291条运动对应的50,004条分段标注，已与官网原始JSON逐项核对一致。该结论针对标签来源一致性，不代表动态物理筛选或完整生成器训练已经完成。**

## 下载与保存

- 官网下载页： https://babel.is.tue.mpg.de/download.php ，本次实际可进入下载区。
- 原始包： https://babel.is.tue.mpg.de/media/upload/data/babel_v1-0_release.zip ，{acquisition['archive_bytes']:,}字节；ZIP CRC及5个JSON解析均通过。
- 原始ZIP SHA256：`{acquisition['archive_sha256']}`。
- 本地原始ZIP、train/val/test及extra_train/extra_val：`D:/具身智能/TextOp复现_20261024/datasets/BABEL`。
- 服务器原始包、5个JSON及私有核查索引：`{install['server_directory']}`，合计{install['bytes']:,}字节，7个文件逐一与本地SHA256一致。目录权限700，文件权限600。
- BABEL许可已获用户独立同意。本次未保存账号密码、未导出浏览器Cookie，原始标注和逐条索引不上传GitHub。

| 文件 | 运动记录 | 序列标签实例 | 帧级标签实例 |
| --- | ---: | ---: | ---: |
'''
    for x in sorted(acquisition['files'],key=lambda x:x['file']):
        report+=f"| {x['file']} | {x['sequence_records']:,} | {x['sequence_labels']:,} | {x['frame_labels']:,} |\n"
    report+='''
extra文件包含多位标注者的补充标注，上表统计标签实例，不能直接相加当作不同运动数量。test与extra文件保留原样，未加入本次训练/验证对齐索引。

## 与发布包的独立比对

| 发布划分 | 运动条数 | 分段标签 | 全项核对一致 |
| --- | ---: | ---: | ---: |
'''
    for split,c in audit['split_counts'].items():
        report+=f"| {split} | {c['sequences']:,} | {c['original_segment_labels']:,} | {c['validated_sequences']:,} |\n"
    report+=f'''
核对包括BABEL ID、原始train/val归属、来源路径、proc_label文本、动作类别列表、分段起止时间与标签数量；差异运动为0，最大起止时间差为{audit['max_observed_time_error_seconds']}秒，数值容差1e-6秒。BABEL原始train/val ID交叉数为0。对序列级备用标注，按官方pack_dataset.py语义使用[0,dur]时段；帧级标注保留原始时段。

第一次直接比较完整路径时有244条不一致，均来自MPIHDM05和DFaust67的外层目录。固定官方pack_dataset.process_babel_json会去掉首层目录，再查找运动文件。本次按同一明确规则处理后全部一致；没有使用模糊匹配或修改文本/时间掩盖差异。初次核查摘要仍保留。

核查后的私有索引：`textop_original_verified_pairs.private.json`，SHA256 `{audit['verified_pair_index_sha256']}`。已有候选运动继续保留原结果，未重写正在运行的GMR输入/输出；本次核查证实其发布包来源标签与原始BABEL一致。

## 剩余工作

BABEL原始标签获取这一阻塞已解除。仍需完成剩余AMASS输入、29DoF GMR重建，并单独检验SMPL-X导出与时间轴的对应情况；标签相同不能代替运动相位检查。论文要求的特权策略4万步训练、宽松/严格两轮动态物理筛选、最终69D语料归一化和加载集成尚未完成。VAE20万步和LDM30万步完整训练尚未启动。

原始标注、私有配对索引、原始/重定向动作和人体模型按许可保存在个人目录。GitHub仅保存代码、哈希、聚合核查结果及本报告。PhyGile、ReactiveBFM保持暂停；其数据集、检查点与实验记录未改动。
'''
    (BASE/'reports/babel_original_alignment_20261025.md').write_text(report,encoding='utf-8')
    (LOCAL/'BABEL原始标注与配对核查_20261025.md').write_text(report,encoding='utf-8')
    page='<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>BABEL原始标注核查</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1000px;margin:32px auto;padding:24px;background:#f6f8fa}pre{white-space:pre-wrap;background:white;border:1px solid #ccc;padding:24px;font:inherit}a{color:#0969da}</style><h1>BABEL原始标注与TextOp配对核查</h1><p><a href="数据准备核查报告_20261025.html">返回数据准备记录</a></p><pre>'+html.escape(report)+'</pre></html>'
    (LOCAL/'BABEL原始标注与配对核查_20261025.html').write_text(page,encoding='utf-8')
    print(json.dumps({'babel_downloaded':True,'server_files_verified':7,'aligned_release_sequences':8291,'aligned_segment_labels':50004,'differences':0,'full_training_started':False}),flush=True)

if __name__=='__main__':main()
