"""Export two independently audited completed samples without waiting for the rest."""
from pathlib import Path
import export_ssm_new6_20hz as exporter
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
exporter.RUN=ROOT/'ssm_new6_completed2_20261101'
exporter.HUMAN=ROOT/'ssm_new6_human_20261101'
exporter.HERE=Path(__file__).resolve().parent
exporter.IDS=('000162','004028')
if __name__=='__main__':exporter.main()
