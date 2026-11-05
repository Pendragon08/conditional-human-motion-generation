"""Audit two independently exported previews while the rest await resources."""
from pathlib import Path
import audit_ssm_new6_20hz as audit
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
audit.RUN=ROOT/'ssm_new6_completed2_20261101'
audit.HUMAN=ROOT/'ssm_new6_human_20261101'
audit.HERE=Path(__file__).resolve().parent
if __name__=='__main__':audit.main()
