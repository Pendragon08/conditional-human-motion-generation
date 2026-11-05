"""Compare two explicit interpretations of HumanML caption interval tags."""
import collections
import json
import math
from pathlib import Path
import zipfile

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
IDS=BASE/'ssm_id_alignment_20261030/ids.jsonl'
TEXT=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
OUT=BASE/'ssm_caption_timebase_audit_20261030'

def main():
    ids=list(map(json.loads,IDS.open(encoding='utf8'))); rows=[];counts=collections.Counter()
    with zipfile.ZipFile(TEXT) as z:
        for clip in ids:
            mid=clip['motion_id']; member='texts/'+mid+'.txt'
            raw=z.read(member).decode('utf8').splitlines()
            n=clip['stop']-clip['start']; legacy_rate=clip['recorded_effective_fps']
            for lineno,line in enumerate(raw,1):
                if not line.strip():continue
                fields=line.split('#')
                if len(fields)!=4:counts['invalid_fields']+=1;continue
                try:start,end=float(fields[2]),float(fields[3])
                except ValueError:counts['invalid_tag']+=1;continue
                if not math.isfinite(start) or not math.isfinite(end):counts['nonfinite_tag']+=1;continue
                if start==0 and end==0:counts['whole_clip']+=1;continue
                if not (0<=start<end):counts['invalid_interval']+=1;continue
                lo,hi=int(start*20),int(end*20)
                legacy_in_bounds=0<=lo<hi<=n
                # The prior loader treats the tag as nominal 20-Hz indices;
                # translate those indices to elapsed time under verified FPS.
                first=lo/legacy_rate;last=hi/legacy_rate
                duration=(n-1)/legacy_rate
                row={'id':mid,'caption_line':lineno,'legacy_rate':legacy_rate,
                     'tag_start_s':start,'tag_end_s':end,
                     'legacy_index_first':lo,'legacy_index_stop':hi,'legacy_index_in_bounds':legacy_in_bounds,
                     'legacy_index_start_actual_s':first,'legacy_index_stop_actual_s':last,
                     'clip_last_frame_time_s':duration,
                     'physical_seconds_start_in_bounds':start<=duration,
                     'physical_seconds_end_within_last_frame':end<=duration,
                     'start_time_interpretation_gap_s':abs(start-first),'end_time_interpretation_gap_s':abs(end-last)}
                rows.append(row);counts['timed_caption']+=1
                if not legacy_in_bounds:counts['legacy_index_out_of_bounds']+=1
                if end>duration:counts['physical_end_beyond_last_frame']+=1
    OUT.mkdir(mode=0o700)
    with (OUT/'intervals.jsonl').open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    rates=collections.defaultdict(list)
    for row in rows:rates['near20' if abs(row['legacy_rate']-20)<0.01 else 'near30'].append(row)
    grouped={k:{'intervals':len(v),'max_start_gap_s':max(x['start_time_interpretation_gap_s'] for x in v),
                'max_end_gap_s':max(x['end_time_interpretation_gap_s'] for x in v),
                'legacy_out_of_bounds':sum(not x['legacy_index_in_bounds'] for x in v),
                'physical_end_beyond_last_frame':sum(not x['physical_seconds_end_within_last_frame'] for x in v)}
             for k,v in rates.items()}
    summary={'counts':dict(counts),'groups':grouped,
       'limitations':['Compares tag-as-physical-seconds and prior loader nominal-index semantics; does not establish which is original annotation intent.',
                      'Stop indices are exclusive and time past last sampled point is reported separately.',
                      'No captions were modified and no output training dataset was created.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
