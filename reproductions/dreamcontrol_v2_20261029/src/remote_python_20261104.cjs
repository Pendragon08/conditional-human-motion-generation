const fs = require('fs');
const cp = require('child_process');
const file = process.argv[2];
const result = cp.spawnSync('ssh', ['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32',process.argv[3]||'python3','-'], {input:fs.readFileSync(file),encoding:'utf8',maxBuffer:8*1024*1024});
process.stdout.write(result.stdout||''); process.stderr.write(result.stderr||'');
process.exit(result.status===null?1:result.status);
