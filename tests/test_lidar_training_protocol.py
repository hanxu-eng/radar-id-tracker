import json
from pathlib import Path
import tempfile
import unittest
from radar_id_tracker.lidar_training_protocol import load_training_plan


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).parents[1]/'configs/lidar_training_coverage.experimental.json').read_text())

    def load(self, plan):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'split.json'; path.write_text(json.dumps(plan))
            return load_training_plan(path)

    def test_actual_plan_reserves_future_sequence_and_has_whole_sequence_roles(self):
        plan,digest=self.load(self.plan)
        self.assertEqual(len(digest),64)
        self.assertEqual(plan['validation_sequences'],['0010','0015','0019'])
        self.assertEqual(plan['reserved_sequences'],['0020'])
        self.assertEqual(len(plan['training_sequences']),17)

    def test_previously_viewed_cannot_be_reclassified_as_unseen(self):
        plan=dict(self.plan, training_sequences=[s for s in self.plan['training_sequences'] if s!='0013'],
                  validation_sequences=['0013'])
        with self.assertRaisesRegex(ValueError,'previously viewed'):self.load(plan)

    def test_rejects_overlap_duplicates_and_malformed_sequence_ids(self):
        for values in (['0020'],['0010','0010'],['../x'],['0021'],'0010'):
            with self.assertRaises(ValueError):self.load(dict(self.plan,validation_sequences=values))

    def test_rejects_unfrozen_selection_and_invalid_guard(self):
        with self.assertRaises(ValueError):self.load(dict(self.plan,validation_used_for_selection=True))
        for value in (float('nan'),0,1.1,True):
            with self.assertRaises(ValueError):self.load(dict(self.plan,acceptance_recall=value))


if __name__=='__main__':unittest.main()
