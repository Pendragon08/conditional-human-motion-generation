import unittest
from provenance_groups import audit_provenance


class ProvenanceTests(unittest.TestCase):
    def test_transitive_crossing(self):
        records = [
            {'sample_id':'a','split':'train','provenance_keys':{'session':['s1']}},
            {'sample_id':'b','split':'train','provenance_keys':{'session':['s1'],'hash':['h']}},
            {'sample_id':'c','split':'test','provenance_keys':{'hash':['h']}},
        ]
        result = audit_provenance(records)
        self.assertFalse(result['source_groups_disjoint'])
        self.assertEqual(result['component_count'], 1)
        self.assertEqual(len(result['cross_split_components'][0]), 3)

    def test_namespaces_and_incomplete_records(self):
        rows = [
            {'sample_id':'a','split':'train','provenance_keys':{'session':['x']}},
            {'sample_id':'b','split':'test','provenance_keys':{'subject':['x']}},
        ]
        self.assertTrue(audit_provenance(rows)['source_groups_disjoint'])
        rows[0]['provenance_keys'] = {}
        with self.assertRaises(ValueError): audit_provenance(rows)

    def test_duplicates_and_bad_split(self):
        row = {'sample_id':'a','split':'train','provenance_keys':{'source':['x']}}
        with self.assertRaises(ValueError): audit_provenance([row,row])
        with self.assertRaises(ValueError): audit_provenance([{**row,'split':'other'}])


if __name__ == '__main__': unittest.main()
