import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from omomo_wrist_box_review_index import distances


class EnclosureTests(unittest.TestCase):
    def test_translation_and_orientation(self):
        lo=np.array([-1.,-2.,-3.]); hi=np.array([1.,2.,3.])
        obj=np.array([[1.,0,0,0, 4,5,6],[2**-.5,0,0,2**-.5,4,5,6]])
        points=np.zeros((2,22,3))
        points[0,20]=[6,5,6];points[0,21]=[4,5,6]
        points[1,20]=[4,7,6];points[1,21]=[4,5,6]
        np.testing.assert_allclose(distances(points,obj,lo,hi),[[1,0],[1,0]],atol=1e-12)

    def test_invalid_object_quaternion_rejected(self):
        with self.assertRaisesRegex(ValueError,'quaternion'):
            distances(np.zeros((2,22,3)),np.zeros((2,7)),np.zeros(3),np.ones(3))


if __name__=='__main__':unittest.main()
