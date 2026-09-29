import unittest
from itertools import product
from random import Random

from radar_id_tracker.assignment import assign


class AssignmentTests(unittest.TestCase):
    def test_global_optimum_and_forbidden_edges(self):
        self.assertEqual(assign([[0.9, 0.2], [0.1, None]], 2), [(0, 1), (1, 0)])

    def test_unmatched_track_does_not_steal_detection(self):
        self.assertEqual(assign([[None], [0.3]], 1), [(1, 0)])

    def test_no_detections(self):
        self.assertEqual(assign([[], []], 0), [])

    def test_small_random_matrices_match_brute_force(self):
        rng = Random(7)
        unmatched_cost = 1.01
        for rows in range(1, 5):
            for columns in range(1, 5):
                for _ in range(20):
                    costs = [
                        [None if rng.random() < 0.25 else rng.random() for _ in range(columns)]
                        for _ in range(rows)
                    ]
                    actual = assign(costs, columns, unmatched_cost)
                    actual_cost = sum(costs[row][col] for row, col in actual)
                    actual_cost += unmatched_cost * (rows - len(actual))
                    possibilities = []
                    for choices in product(range(-1, columns), repeat=rows):
                        selected = [col for col in choices if col >= 0]
                        if len(selected) != len(set(selected)):
                            continue
                        if any(col >= 0 and costs[row][col] is None for row, col in enumerate(choices)):
                            continue
                        possibilities.append(
                            sum(unmatched_cost if col < 0 else costs[row][col]
                                for row, col in enumerate(choices))
                        )
                    self.assertAlmostEqual(actual_cost, min(possibilities), places=8)


if __name__ == "__main__":
    unittest.main()
