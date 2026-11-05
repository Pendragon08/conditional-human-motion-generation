import numpy as np
import pytest
from humanml_joint_mirroring import amass_to_humanml,mirror_humanml

@pytest.mark.parametrize('joints',[22,52])
def test_axes_and_mirror_involution(joints):
    a=np.arange(3*2*joints,dtype=np.float64).reshape(2,joints,3)
    converted=amass_to_humanml(a)
    np.testing.assert_array_equal(converted[...,0],-a[...,0])
    np.testing.assert_array_equal(converted[...,1],a[...,2])
    np.testing.assert_array_equal(converted[...,2],a[...,1])
    mirrored=mirror_humanml(converted)
    np.testing.assert_array_equal(mirrored[:,1,0],-converted[:,2,0])
    np.testing.assert_array_equal(mirrored[:,1,1:],converted[:,2,1:])
    np.testing.assert_array_equal(mirror_humanml(mirrored),converted)
    np.testing.assert_array_equal(a,np.arange(3*2*joints,dtype=np.float64).reshape(2,joints,3))

def test_rejects_bad_shapes_and_nonfinite():
    for a in (np.zeros((3,25,3)),np.zeros((3,22,2)),np.full((2,22,3),np.nan)):
        with pytest.raises(ValueError):mirror_humanml(a)
