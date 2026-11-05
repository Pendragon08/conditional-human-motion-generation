// Stage the pinned, inference-only conversion of the original HumanML3D evaluator.
const cp = require('node:child_process');
const crypto = require('node:crypto');

const base = 'https://huggingface.co/ZeyuLing/motius-evaluator-humanml3d-official/resolve/d6f2735efeb5319ce015e1c2247b636685fe457a/';
const host = '10.147.17.32';
const root = '/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/generated_reference_evaluation_20261105/evaluator_assets_v2';
const members = {
  'model.safetensors': [86766520, '569bbd5526cb4b5fdda48a08abf7dd0b5a1e9a39ea0be381730771ac18063981'],
  'stats/mean.npy': [2232, '0bdb5ba69a3a9e34d71990db15bc535ebc024c8d95ddb5574196f96058faa7d3'],
  'stats/std.npy': [2232, '6a5f7d60301c9465972fc225f8ad0ee8f957e7720431189123eb6d15873a9557'],
  'glove/our_vab_data.npy': [10077728, '73ed9b73ee5bc5ab683661c132b9f98fe305672dffdf70ceeac1bc447543d668'],
  'glove/our_vab_idx.pkl': [79811, 'd85dce83d2c27a92bb94bb51dc1f55a04fcaa328c359d23eccf648e296c16493'],
  'glove/our_vab_words.pkl': [67470, 'fac88da1be6f00d36b72b88c9745c8ac35a94a7bd6ccf1fa3bab380faf8c2e0d'],
};

const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
const fs = require('node:fs');
const staging = 'D:/DreamControl_eval_tokenizer_20261105/evaluator-transfer';
fs.mkdirSync(staging, {recursive:true});

function checked(command, args, options = {}) {
  const result = cp.spawnSync(command, args, {timeout: 360000, windowsHide: true, ...options});
  if (result.status !== 0) throw new Error(result.error?.message || result.stderr.toString());
  return result.stdout?.toString().trim();
}

function upload(name, data, digest) {
  const path = root + '/' + name;
  const partial = path + '.incoming';
  const local = staging + '/' + digest;
  const ssh = ['-o','BatchMode=yes','-o','ConnectTimeout=15'];
  const config = [
    'import pathlib,hashlib,sys,shutil',
    'p=pathlib.Path(' + JSON.stringify(path) + ')',
    'n=' + data.length,
    'assert shutil.disk_usage("/data").free-n>=8*1024**3+456*1024**2',
    'p.parent.mkdir(parents=True,exist_ok=True)',
    'if p.exists(): assert hashlib.sha256(p.read_bytes()).hexdigest()==' + JSON.stringify(digest),
    'print("present" if p.exists() else "stage")',
  ].join('\n');
  if (checked('ssh', [...ssh,host,'python3 -c '+quote(config)]) === 'present') return;
  if (!fs.existsSync(local)) {
    const disk=fs.statfsSync(staging);
    if (disk.bavail * disk.bsize - data.length < 1024**3 + 100*1024**2) throw new Error('Local staging floor');
    fs.writeFileSync(local,data,{flag:'wx'});
  }
  if (sha(fs.readFileSync(local)) !== digest) throw new Error('Existing staging digest mismatch');
  checked('scp', ['-q',...ssh,local,host+':'+partial], {timeout: 600000});
  const verify = [
    'import pathlib,hashlib',
    'p=pathlib.Path(' + JSON.stringify(path) + ')',
    'q=pathlib.Path(' + JSON.stringify(partial) + ')',
    'assert not p.exists() and not q.is_symlink()',
    'assert q.stat().st_size==' + data.length,
    'assert hashlib.sha256(q.read_bytes()).hexdigest()==' + JSON.stringify(digest),
    'q.rename(p)',
  ].join('\n');
  checked('ssh', [...ssh,host,'python3 -c '+quote(verify)]);
  fs.unlinkSync(local);
}
const receipt = [];
for (const [member, [size, digest]] of Object.entries(members)) {
  const source = base + member;
  const result = cp.spawnSync('curl.exe', [
    '--fail','--silent','--show-error','--location','--retry','2','--max-time','180',
    '--max-filesize',String(size),source,
  ], {maxBuffer: size + 65536, timeout: 240000, windowsHide: true});
  if (result.status !== 0) throw new Error(result.error?.message || result.stderr.toString());
  if (result.stdout.length !== size || sha(result.stdout) !== digest) throw new Error('Source artifact digest mismatch: ' + member);
  upload(member, result.stdout, digest);
  receipt.push({member, size, sha256: digest, source});
  console.log(member, digest);
}
const manifest = Buffer.from(JSON.stringify(receipt, null, 2) + '\n');
upload('source_assets.json', manifest, sha(manifest));