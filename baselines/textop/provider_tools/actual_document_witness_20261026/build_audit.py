import pathlib
W=pathlib.Path(__file__).resolve().parent
s=(W/'preflight_remote_v2.py').read_text(encoding='utf-8')
s=s.replace("run='quick_pilot_smoke_author_v1'","run='quick_pilot_smoke_document_v1'")
start=s.index("newrun='quick_pilot_smoke_document_v1';newprefix=")
end=s.index("report=",start)
s=s[:start]+s[end:]
s=s.replace("'status':'PREFLIGHT_PASSED'","'status':'DOCUMENT_ACTUAL_AUDIT_PASSED'")
s=s.replace("'author':data","'document':data").replace("'author_hashes':","'document_hashes':").replace("'author_private_verified':","'document_private_verified':").replace("'fresh_document_outputs_absent':True","'document_outputs_exist':True")
marker="report={"
extra=r'''
# Read private binary and NPZ payloads in place using stdlib only. No SDK or package imports.
import ast,struct,zipfile,math
def npy(payload):
    assert payload[:6]==b'\x93NUMPY'
    major=payload[6]
    if major==1:
        length=struct.unpack('<H',payload[8:10])[0];offset=10
    else:
        assert major in [2,3]
        length=struct.unpack('<I',payload[8:12])[0];offset=12
    header=ast.literal_eval(payload[offset:offset+length].decode('utf-8').strip())
    assert header['fortran_order'] is False
    return header,payload[offset+length:]
payload_count=0;binary_bytes=npz_bytes=all_samples=0;sample_lengths=[];per_trial_steps=[]
for record in records:
    folder=pathlib.Path(record['folder']);meta=json.loads((folder/'trace.private.json').read_text())
    binary=folder/'poses.private.bin';compressed=folder/'executed_poses.private.npz'
    raw=binary.read_bytes();count=meta['samples'];width=meta['row_width']
    assert len(raw)==count*width*4
    floats=struct.unpack('<'+'f'*(count*width),raw)
    assert all(math.isfinite(v) for v in floats)
    with zipfile.ZipFile(compressed) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={f['name']+'.npy' for f in meta['fields']}|{'time_seconds.npy','joint_names.npy','runtime_kind.npy','capture_phase.npy'}
        offset=0
        for field in meta['fields']:
            h,payload=npy(z.read(field['name']+'.npy'))
            shape=tuple(field['shape']);size=math.prod(shape)
            assert h['descr']=='<f4' and h['shape']==(count,)+shape
            values=struct.unpack('<'+'f'*(count*size),payload)
            expected=tuple(floats[i*width+offset+j] for i in range(count) for j in range(size))
            assert values==expected
            if field['name']=='reference_frame':
                assert values==(0.0,)+tuple(float(i) for i in range(count-1))
            if field['name']=='root_quat_wxyz':
                assert all(.99<math.sqrt(sum(values[i*4+j]**2 for j in range(4)))<1.01 for i in range(count))
            offset+=size;payload_count+=1
        assert offset==width
        h,payload=npy(z.read('time_seconds.npy'))
        assert h['descr']=='<f8' and h['shape']==(count,)
        times=struct.unpack('<'+'d'*count,payload)
        assert times==tuple(i/50. for i in range(count))
        for name,expected in [('runtime_kind',meta['runtime_kind']),('capture_phase',meta['phase'])]:
            h,payload=npy(z.read(name+'.npy'));assert h['shape']==() and h['descr'].startswith('<U')
            assert payload.decode('utf-32-le').rstrip('\0')==expected
        h,payload=npy(z.read('joint_names.npy'))
        assert h['shape']==(29,) and h['descr'].startswith('<U')
        size=int(h['descr'][2:])*4
        names=[payload[i*size:(i+1)*size].decode('utf-32-le').rstrip('\0') for i in range(29)]
        assert names==meta['joint_names']
    assert meta['final_reference_endpoint_comparison_performed'] is False
    assert meta['consumed_reference_frame_max']==meta['trial']['control_steps']-1
    assert meta['reference_final_frame']==meta['reference_frames']-1
    assert meta['samples']==meta['reference_frames'] if meta['trial']['complete'] else meta['samples']<=meta['reference_frames']
    all_samples+=count;binary_bytes+=len(raw);npz_bytes+=compressed.stat().st_size
    sample_lengths.append(count);per_trial_steps.append(meta['trial']['control_steps'])
assert len(records)==r['finished_trials']==8 and r['completion_branch_exercised'] is True
assert max(per_trial_steps)==r['actual_vectorized_control_calls']
assert binary_bytes+npz_bytes==r['private_trace_bytes']
assert not list(out.rglob('*.partial'))
open_private_fds=[]
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        if proc.stat().st_uid!=os.getuid():continue
        for fd in (proc/'fd').iterdir():
            try:
                target=os.readlink(fd)
                if target.startswith(str(out)+'/'):open_private_fds.append({'pid':int(proc.name),'fd':fd.name})
            except OSError:pass
    except (OSError,ProcessLookupError):pass
assert not open_private_fds
private_payload_audit={'npz_binary_arrays_exact':True,'finite_pose_action_and_target_values':True,'pre_autoreset_capture_phase_verified':True,'initial_then_consumed_reference_sequence_verified':True,'time_seconds_exact_50hz':True,'joint_names_exact':True,'npz_crc_passed':True,'payload_array_count':payload_count,'closed_trace_count':len(records),'total_samples':all_samples,'sample_lengths_sorted':sorted(sample_lengths),'per_trial_control_steps_sorted':sorted(per_trial_steps),'private_binary_bytes':binary_bytes,'private_npz_bytes':npz_bytes,'partial_files':0,'open_owned_private_fds':open_private_fds,'private_arrays_downloaded':False,'final_reference_endpoint_comparison_performed':False}
'''
# Source literal must contain actual magic bytes notation, not a doubled escape.
extra=extra.replace(r"b'\\x93NUMPY'",r"b'\x93NUMPY'").replace(r"rstrip('\\0')",r"rstrip('\0')")
s=s.replace(marker,extra+"\n"+marker,1)
s=s.replace("'document_outputs_exist':True}","'document_outputs_exist':True,'private_payload_audit':private_payload_audit}")
(W/'audit_remote.py').write_text(s,encoding='utf-8')
