from pathlib import Path
import csv,hashlib,json,math,zipfile
r=Path(__file__).resolve().parents[1]
root=r/'cache/humanml3d_public_20261018';out=r/'results/paired_data_20261018'
m=json.loads((out/'public_metadata_manifest.json').read_text(encoding='utf-8'))
splits={n:set((root/f'HumanML3D/{n}.txt').read_text().split()) for n in ('all','train','val','test','train_val')}
index=list(csv.DictReader((root/'index.csv').open(encoding='utf-8-sig',newline='')))
index_names={Path(x['new_name']).stem for x in index}
source_by_id={Path(x['new_name']).stem:x['source_path'] for x in index}
canonical={n:{x.removeprefix('M') for x in v} for n,v in splits.items()}
source_groups={n:{source_by_id[x] for x in v} for n,v in canonical.items()}
source_overlaps={a+'_'+b:sorted(source_groups[a]&source_groups[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
bad=[];caption_ids=set();finite_segment_count=0;whole_count=0;total=0
with zipfile.ZipFile(root/'HumanML3D/texts.zip') as z:
 assert z.testzip() is None
 for info in z.infolist():
  p=Path(info.filename)
  if p.suffix!='.txt' or p.name.startswith('._') or '__MACOSX' in p.parts:continue
  caption_ids.add(p.stem)
  for no,line in enumerate(z.read(info).decode('utf-8-sig').splitlines(),1):
   if not line.strip():continue
   total+=1;parts=line.split('#');issue=None
   if len(parts)!=4:issue='field_count'
   else:
    try:
     start,end=map(float,parts[2:])
     if not math.isfinite(start) or not math.isfinite(end):issue='nonfinite_time_tags'
     elif start==end==0:whole_count+=1
     elif start<0 or end<=start:issue='invalid_time_interval'
     else:finite_segment_count+=1
    except ValueError:issue='unparseable_time_tags'
   if issue:bad.append({'sample_id':p.stem,'line_number':no,'issue':issue})
m['nonzero_time_tag_captions']=finite_segment_count
m['invalid_caption_lines']=bad
m['caption_inspection_note']='Nonfinite time tags are listed as issues; they are not counted as valid segmented captions. Raw captions are unchanged.'
(out/'public_metadata_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result={'status':'PUBLIC_METADATA_ONLY_NOT_TRAINING_READY','zip_crc_pass':True,
 'text_files':len(caption_ids),'caption_lines':total,'whole_motion_caption_lines':whole_count,
 'finite_segment_caption_lines':finite_segment_count,'time_tag_issues':bad,
 'split_union_equals_all':splits['train']|splits['val']|splits['test']==splits['all'],
 'train_val_equals_train_union_val':splits['train_val']==splits['train']|splits['val'],
 'canonical_motion_id_overlaps':{a+'_'+b:sorted(canonical[a]&canonical[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
 'raw_source_path_overlap_counts':{k:len(v) for k,v in source_overlaps.items()},
 'raw_source_path_overlap_examples':{k:v[:10] for k,v in source_overlaps.items()},
 'official_split_ids_missing_text':sorted(splits['all']-caption_ids),
 'text_ids_not_in_official_all_split':sorted(caption_ids-splits['all']),
 'official_all_ids_without_index_mapping':sorted(canonical['all']-index_names),
 'paired_robot_motion_sequences':0,'generator_training_started':False,
 'public_metadata_manifest_sha256':hashlib.sha256((out/'public_metadata_manifest.json').read_bytes()).hexdigest()}
assert result['split_union_equals_all'] and result['train_val_equals_train_union_val']
assert not result['official_split_ids_missing_text'] and not result['official_all_ids_without_index_mapping']
assert not any(result['canonical_motion_id_overlaps'].values())
(out/'alignment_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''# PhyGile 配对数据准备状态（2026-10-18）

**已取得官方公开文本、划分和来源索引；尚未取得对应 G1 动作训练集，配对机器人序列数为 0，未开始生成器训练。** 此状态是上轮“缺配对数据”的细化，不表示 HumanML3D 没有文本标注。

## 本次实际取得

固定官方 HumanML3D 提交 `{m['pinned_commit']}`，下载 README、LICENSE、index.csv、texts.zip 与五份划分文件，共 9 个文件，逐文件 SHA-256 见 `results/paired_data_20261018/public_metadata_manifest.json`。

- texts.zip：{len(caption_ids):,} 个文本文件，{total:,} 条非空标注。
- 来源索引：{m['source_index_rows']:,} 行，字段为 source_path、start_frame、end_frame、new_name。
- 实际官方划分：train {len(splits['train']):,}、val {len(splits['val']):,}、test {len(splits['test']):,}，合计 {len(splits['all']):,} 个 ID。
- 文本包比 all.txt 多 {len(result['text_ids_not_in_official_all_split'])} 个 ID，按官方划分处理，不把额外条目静默并入训练。
- ZIP CRC、划分并集、文本覆盖、原始索引覆盖及去镜像后的动作 ID 隔离均核验通过。时间标签异常条数 {len(bad)}，逐行位置保留在 `alignment_inventory.json`，需要在后续配对时显式处理，尚未据此生成任何动作。
- 按 index.csv 的 source_path 核对，原始来源文件跨划分重叠数：{result['raw_source_path_overlap_counts']}。动作 ID 隔离不等同于受试者隔离；不能据此声称未见受试者泛化。

## 论文所需的配对关系

PhyGile §III-B 使用经过筛选、重定向到机器人形态的 HumanML3D 动作与原文本配对。文本无须全部重新标注。需要按同一 ID、原始录制、时间范围和镜像关系处理：

`HumanML3D 文本及索引 + 对应 AMASS 动作 → GMR 重定向到 G1 → 时间/语义与运动学核对 → 本实现的 262D 编码 → 训练 manifest`

保留官方 train/val/test 与镜像对应关系，并核对原始录制来源；子片段标注必须按时间对齐，不能把整句标签赋给任意裁剪片段。HumanML3D 的人体特征不能通过删改一列直接作为 G1 262D 数据。

官方动作 ID 划分存在上述原始来源文件重叠，不能同时宣称“原样官方划分”和“原始录制文件完全隔离”。现有训练入口会拒绝 source_group 跨划分重叠，后续必须明确分组与实验协议；若改变官方划分，应记录与原论文的差异。本次仅准备元数据，没有据此创建或启动研究训练。

## 当前具体缺什么

1. 与索引对应的原始 AMASS 动作文件。AMASS 官网下载页需要账号登录。
2. 所选重定向路径所需的人体模型。当前官方 GMR 的 AMASS 路线使用 SMPL-X 动作和 SMPL-X 模型文件，并明确提示不要直接使用 SMPL+H 输入。SMPL-X 官网要求注册并同意许可。
3. 实际机器人动作生成、时间对齐、过滤与 262D 提取；这些步骤尚未在本次公开文本上执行，不能把文本下载当作完整配对数据已就绪。

HumanML3D 官方原始重建脚本使用 SMPL+H/DMPL；它与 GMR 的 SMPL-X 入口不是同一格式，需选择并核对一致的处理链。没有把第三方镜像冒充作者发布的 G1 数据包。本次检查 PhyGile 项目及跟踪仓库，未找到作者筛选后的 G1 动作—文本配对包下载入口。

当前 20/4 个武术参考没有对应的 HumanML3D 原始文本，仍用于跟踪验证，不当作该生成器的完整配对训练集。

## 来源

- [PhyGile 原文 §III-B](https://arxiv.org/html/2603.19305v3)
- [PhyGile 官方项目](https://baojch.github.io/phygile-page/)
- [HumanML3D 官方获取说明](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data)
- [HumanML3D 官方文本及划分目录](https://github.com/EricGuo5513/HumanML3D/tree/{m['pinned_commit']}/HumanML3D)
- [AMASS 下载入口](https://amass.is.tue.mpg.de/download.php)
- [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/)
- [GMR Data Preparation](https://github.com/YanjieZe/GMR#data-preparation)
'''
(r/'docs/paired_data_status_20261018.md').write_text(text,encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('time_tag_issues','canonical_motion_id_overlaps','raw_source_path_overlap_examples')},ensure_ascii=False,indent=2))
print('time_tag_issues',len(bad),flush=True)
