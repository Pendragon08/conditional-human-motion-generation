const fs=require('node:fs');
const file='C:/Users/Pende/.codex/automations/dreamcontrol-v2/automation.toml';
const raw=fs.readFileSync(file,'utf8');const fields={};
for(const line of raw.split(/\r?\n/)){const m=line.match(/^([a-z_]+)\s*=\s*(.+)$/);if(m){try{fields[m[1]]=JSON.parse(m[2]);}catch{fields[m[1]]=m[2];}}}
process.stdout.write(JSON.stringify(fields));
