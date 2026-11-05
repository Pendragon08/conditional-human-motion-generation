"""Use one previously audited SSM run to test the new exporter in a temp dir."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import export_ssm_new6_resume_20hz as module

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')

class IntegrationTest(unittest.TestCase):
    def test_existing_audited_single_clip(self):
        oldrun=ROOT/'ssm_rawfps_anchor_gpu_20261031';oldhuman=ROOT/'ssm_rawfps_human_batch_20261031'
        prior=ROOT/'ssm_anchored_20hz_candidates_20261031/007410.npz'
        with tempfile.TemporaryDirectory(prefix='ssm_export_check_') as d:
            root=Path(d);run=root/'run';out=root/'out';run.mkdir();out.mkdir()
            (run/'007410').symlink_to(oldrun/'007410',target_is_directory=True)
            (run/'queue_status.txt').write_text('audited\n')
            source=json.loads((oldrun/'batch_audit.json').read_text())
            row=next(x for x in source['cases'] if x['id']=='007410')
            (run/'batch_audit.json').write_text(json.dumps({'cases':[row]}))
            saved=(module.RUN,module.HUMAN,module.HERE,module.IDS)
            try:
                module.RUN=run;module.HUMAN=oldhuman;module.HERE=out;module.IDS=('007410',)
                module.main()
            finally:module.RUN,module.HUMAN,module.HERE,module.IDS=saved
            report=json.loads((out/'report.json').read_text())
            self.assertEqual((report['successful'],report['failed']),(1,[]))
            with np.load(out/'007410.npz',allow_pickle=False) as now,np.load(prior,allow_pickle=False) as before:
                np.testing.assert_array_equal(now['features'],before['features'])
                np.testing.assert_array_equal(now['sample_times'],before['sample_times'])
                np.testing.assert_array_equal(now['joint_pos'],before['joint_pos'])

if __name__=='__main__':unittest.main()
