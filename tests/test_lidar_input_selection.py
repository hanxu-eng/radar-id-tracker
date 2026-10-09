import unittest
from radar_id_tracker.lidar_input_selection import select_grouped_inputs
from radar_id_tracker.lidar_inference_config import RawLidarConfig


class InputSelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(sequence=s,scale=p,sweeps=5,
                     observed=dict(precision=.9,recall=f,f1=f))
                for s,values in [('0000',(.99,.75)),('0001',(.4,.75)),('0005',(.4,.75))]
                for p,f in zip((255,16),values)]

    def test_worst_sequence_counts_and_held_out_does_not_select_itself(self):
        rows=self.rows();result=select_grouped_inputs(rows)
        self.assertEqual(result['selected_inference_config']['intensity_scale'],16)
        changed=[{**r,'observed':dict(precision=1.,recall=1.,f1=1.)} if r['sequence']=='0000' and r['scale']==255 else r for r in rows]
        a=next(f for f in result['folds'] if f['held_out_sequence']=='0000')
        b=next(f for f in select_grouped_inputs(changed)['folds'] if f['held_out_sequence']=='0000')
        self.assertEqual(a['selected_scale'],b['selected_scale'])
        self.assertFalse(result['development_acceptance_passed'])

    def test_incomplete_or_duplicate_grid_is_rejected(self):
        rows=self.rows()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):select_grouped_inputs(bad)

    def test_zero_outputs_cannot_pass_precision_recall_guard(self):
        rows=self.rows()
        for r in rows:
            if r['scale']==255:r['observed']=dict(precision=None,recall=0.,f1=0.)
        result=select_grouped_inputs(rows)
        self.assertEqual(result['selected_inference_config']['intensity_scale'],16)
        self.assertFalse(result['development_acceptance_passed'])

    def test_invalid_inference_config_fails_before_gpu_import(self):
        for kwargs in [dict(sweeps=True),dict(sweeps=11),dict(intensity_scale=0),
                       dict(period_s=float('nan')),dict(score_threshold=1),dict(voxel_backend='other')]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):RawLidarConfig(**kwargs)
        with self.assertRaises(TypeError):RawLidarConfig.from_dict({'typo':1})


if __name__=='__main__':unittest.main()
