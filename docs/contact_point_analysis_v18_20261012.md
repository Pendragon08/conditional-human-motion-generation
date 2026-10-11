# V18 read-only humanoid native-contact diagnosis

Command (already executed once, zero simulation):

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/contact_point_analysis_v18_20261012.py
```

Output:12 inputs,4818 positive-force native ground-contact points during valid generated frames. Body-origin and COM velocity interpretations are reported separately as diagnostic hypotheses, not replacements for the old score. With either interpretation, all12 point-fraction diagnostics remain above.10; this is not a calibrated aggregate success metric.30Hz finite differences alone cannot settle the velocity convention. V19 subsequently supplies an independent240Hz offset-COM calibration. Dense human arrays/manifolds remain server-local; input hashes and compact diagnostic summaries are published. Thresholds, original goals, old results and models unchanged. No Overleaf or innovation claim.
