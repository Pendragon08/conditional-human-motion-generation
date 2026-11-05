import unittest
import numpy as np
from mujoco_contact_flags import confusion


class ContactConfusionTests(unittest.TestCase):
    def test_per_foot_counts_and_metrics(self):
        source=np.array([[1,0],[1,0],[0,1]],dtype=bool)
        target=np.array([[1,1],[0,0],[0,1]],dtype=bool)
        result=confusion(source,target)
        self.assertEqual([result[k] for k in ('tp','fn','fp','tn')],[2,1,1,2])
        self.assertAlmostEqual(result['recall'],2/3)
        self.assertAlmostEqual(result['precision'],2/3)
        self.assertAlmostEqual(result['agreement'],4/6)
    def test_rejects_shape_and_nonboolean(self):
        with self.assertRaises(ValueError): confusion(np.zeros((2,2)),np.zeros((2,2)))
        with self.assertRaises(ValueError): confusion(np.zeros((2,1),bool),np.zeros((2,1),bool))


if __name__=='__main__': unittest.main()
