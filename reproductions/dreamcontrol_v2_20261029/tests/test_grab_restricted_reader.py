import io
import os
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
from grab_restricted_reader import read_object_npy


def encode(value):
    f=io.BytesIO()
    np.lib.format.write_array_header_1_0(f,{'descr':'|O','fortran_order':False,'shape':()})
    f.write(pickle.dumps(np.array(value,dtype=object),protocol=3))
    return f.getvalue()


class Forbidden:
    def __reduce__(self): return os.system,('this_command_must_never_run',)


class RestrictedReaderTests(unittest.TestCase):
    def test_numeric_dictionary_roundtrip(self):
        original={'params':{'transl':np.array([[1.,2.,3.]],dtype=np.float32)},'gender':'male'}
        result=read_object_npy(encode(original))
        np.testing.assert_array_equal(result['params']['transl'],original['params']['transl'])
        self.assertEqual(result['gender'],'male')

    def test_forbidden_global_is_rejected(self):
        with self.assertRaises(pickle.UnpicklingError): read_object_npy(encode({'payload':Forbidden()}))

    def test_invalid_nested_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'finite'): read_object_npy(encode({'x':np.array([float('nan')])}))
        with self.assertRaisesRegex(ValueError,'numeric'): read_object_npy(encode({'x':np.array(['oops'])}))


if __name__=='__main__': unittest.main()
