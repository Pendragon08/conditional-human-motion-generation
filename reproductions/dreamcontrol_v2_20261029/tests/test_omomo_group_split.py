import unittest
from omomo_group_split import classify


class GroupSplitTest(unittest.TestCase):
    def row(self,subject='sub3',number='042',variant='rot_1'):
        key=f'{subject}_largebox_{number}'
        return {'motion_member':f'robot-object/{key}_{variant}.npz','nominal_action_key':key,
                'variant':variant,'provisional_group':f'OMOMO_nominal_subject:{subject}'}

    def test_original_and_variants_share_nominal_group(self):
        self.assertEqual(classify(self.row(variant='original')),('sub3','sub3_largebox_042','original','train'))
        self.assertEqual(classify(self.row(variant='trans_2')),('sub3','sub3_largebox_042','trans_2','train'))
        self.assertEqual(classify(self.row(subject='sub12'))[-1],'val')
        self.assertEqual(classify(self.row(subject='sub16'))[-1],'test')

    def test_disagreeing_metadata_and_unsafe_path_fail(self):
        r=self.row();r['nominal_action_key']='sub3_largebox_043'
        with self.assertRaises(ValueError):classify(r)
        r=self.row();r['provisional_group']='OMOMO_nominal_subject:sub16'
        with self.assertRaises(ValueError):classify(r)
        r=self.row();r['motion_member']='../'+r['motion_member']
        with self.assertRaises(ValueError):classify(r)


if __name__=='__main__':unittest.main()
