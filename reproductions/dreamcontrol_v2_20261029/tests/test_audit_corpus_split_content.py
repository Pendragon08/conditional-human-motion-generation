import pickle
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from audit_corpus_split_content import audit


class ContentSplitTest(unittest.TestCase):
    def test_detects_exact_motion_and_caption_cross_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data={}
            for key,value,text in [('a',0,'Same Caption'),('b',0,' same  caption '),('c',1,'different')]:
                data[key]={'motion':np.full((3,36),value,dtype=np.float32),'text':[{'caption':text}]}
            corpus=root/'trusted.pkl'
            with corpus.open('wb') as f: pickle.dump({'data_dict':data},f)
            for name,content in [('train','a\n'),('val','c\n'),('test','b\n')]:
                (root/(name+'.txt')).write_text(content)
            report=audit(corpus,root)
        self.assertEqual(report['exact_motion_cross_split_groups'],1)
        self.assertEqual(report['identical_caption_cross_split_groups'],1)
        self.assertEqual(report['unassigned_corpus_samples'],0)

    def test_rejects_duplicate_id_across_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); corpus=root/'trusted.pkl'
            with corpus.open('wb') as f:
                pickle.dump({'data_dict':{'a':{'motion':np.zeros((2,36)),'text':[]}}},f)
            for name in ('train','val','test'):
                (root/(name+'.txt')).write_text('a\n' if name!='val' else '')
            with self.assertRaisesRegex(ValueError,'multiple splits'):
                audit(corpus,root)


if __name__ == '__main__':
    unittest.main()
