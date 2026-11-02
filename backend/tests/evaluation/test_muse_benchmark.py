"""Offline deterministic comparison tests: no model calls or credentials."""
import unittest
from app.experiments.muse_benchmark import BaselineSelector, compare


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.worlds = [(0.23, 0.34), (0.17, 0.65)]
        self.selectors = {
            name: (lambda i, name=name: BaselineSelector(name, seed=42+i))
            for name in ('fixed', 'random', 'diverse')
        }

    def test_worlds_and_budgets_match(self):
        result = compare(self.selectors, self.worlds)
        for row in result.values():
            self.assertEqual(row['worlds'], 2)
            self.assertEqual(len(row['errors_m']), 2)
            self.assertEqual([len(x) for x in row['selections_mps']], [2, 2])
            self.assertTrue(all(speed in (2., 4., 6.) for trial in row['selections_mps'] for speed in trial))

    def test_comparison_reproducible(self):
        self.assertEqual(compare(self.selectors, self.worlds), compare(self.selectors, self.worlds))

    def test_empty_worlds_rejected(self):
        with self.assertRaisesRegex(ValueError, 'At least one world'):
            compare(self.selectors, [])


if __name__ == '__main__':
    unittest.main()
