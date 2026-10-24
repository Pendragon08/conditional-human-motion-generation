# Canonical-spec digest document witness

Verdict: PASS

Scope: Independent fresh-agent check of canonical-spec digest documentation only.
Read: compute-env-contract.md, docs/COMPUTE_LEDGER.md, docs/ENVIRONMENT.md.
Working directory: C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree
Shell: C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe
login: false

Exact command:
```powershell
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'scripts/canonical_env_hash.py' 'configs/environment_plan_v2.json'
```

Exit code: 0
Expected SHA-256: a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954
Observed SHA-256: a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954
Observed short digest: a43f960b
Canonical byte count reported by command: 1921

The command reproduced the documented full SHA-256 and short digest. Its output reports JSON parsing, recursively sorted keys, compact separators, ensure_ascii=True, UTF-8 bytes, and no newline, consistent with the ledger.
No bounded physics commands were executed. Existing fixed output paths were left untouched. This witness does not revalidate the environment, physics behavior, GUI viewing, or training.
