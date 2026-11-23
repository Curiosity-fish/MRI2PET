import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from mri2pet.data.paired_mri_pet_clinical import prepare_table_dataframe,PairedT1FDGTableDataset
from mri2pet.losses.generation import PARC98_HOMOLOGOUS_PAIRS,parc98_losses

class IntegrityTests(unittest.TestCase):
    def test_visit_group_never_fills_from_another_person_or_visit(self):
        raw=pd.DataFrame({'PTID':['a','b','a','a','b'],
                          'VISDATE':['2020-01-01','2020-01-01','2020-01-01','2020-02-01','2020-01-01'],
                          'x':[np.nan,999.0,3.0,np.nan,np.nan],
                          'y':[np.nan,77.0,5.0,np.nan,np.nan]})
        with patch('mri2pet.data.paired_mri_pet_clinical.pd.read_csv',return_value=raw):
            result,_=prepare_table_dataframe('ignored.csv',['x','y'])
        by={(r.PTID,r.DATE_KEY):r for r in result.itertuples()}
        self.assertEqual(by[('a','2020-01-01')].x,3.0)
        self.assertTrue(pd.isna(by[('a','2020-02-01')].x))
        self.assertTrue(pd.isna(by[('a','2020-02-01')].y))
        self.assertEqual(by[('b','2020-01-01')].x,999.0)
        ds=object.__new__(PairedT1FDGTableDataset)
        ds.max_table_delta_days=None; ds.table_cols=['x','y']; ds.standardize_table=False
        ds.tabular_stats={'medians':np.array([11.,12.],dtype=np.float32)}
        ds.table_index={(r['PTID'],r['DATE_KEY']):r for _,r in result.iterrows()}
        values,mask=ds._get_table_tensors(pd.Series({'subject_id':'a','table_date_key':'2020-02-01'}))
        np.testing.assert_allclose(values.numpy(),[11.,12.])
        np.testing.assert_allclose(mask.numpy(),[1.,1.])

    def test_static_group_does_not_fill_from_another_person(self):
        raw=pd.DataFrame({'PTID':['a','b','b','c'],'x':[8.,np.nan,4.,np.nan],'y':[6.,np.nan,np.nan,np.nan]})
        with patch('mri2pet.data.paired_mri_pet_clinical.pd.read_csv',return_value=raw):
            result,_=prepare_table_dataframe('ignored.csv',['x','y'])
        by=result.set_index('PTID')
        self.assertEqual(float(by.loc['b','x']),4.)
        self.assertTrue(pd.isna(by.loc['b','y']))
        self.assertTrue(pd.isna(by.loc['c','x']))

    def test_hac_uses_only_true_homologous_pairs(self):
        pairs=PARC98_HOMOLOGOUS_PAIRS
        self.assertEqual(len(pairs),47)
        used={v for pair in pairs for v in pair}
        self.assertEqual(len(used),94)
        self.assertEqual(set(range(1,99))-used,{10,11,12,15})
        self.assertIn((1,18),pairs)
        self.assertIn((13,27),pairs)
        self.assertIn((31,65),pairs)
        labels=torch.arange(1,99).reshape(1,1,98)
        target=torch.zeros(1,1,1,1,98)
        pred=torch.zeros_like(target)
        for i,(left,right) in enumerate(pairs):
            pred[...,left-1]=i+1
            pred[...,right-1]=i+1
        pred[...,9]=200.
        pred[...,10]=300.
        self.assertAlmostEqual(float(parc98_losses(pred,target,labels)['hac']),0.,places=7)
        pred[...,17]=0.
        self.assertGreater(float(parc98_losses(pred,target,labels)['hac']),0.01)

if __name__=='__main__':
    unittest.main()
