import unittest
from math import isclose

import simulation


class SimulationOracleTests(unittest.TestCase):
    def test_oracle_outputs_are_well_formed_and_deterministic(self):
        sample = simulation.OracleSample(
            m1=1.2,
            m2=0.9,
            base_distance=3.0,
            delta1=0.4,
            delta2=0.6,
            interaction_time=1.1,
        )

        out1 = simulation.oracle(sample)
        out2 = simulation.oracle(sample)

        self.assertEqual(out1.branch_distances, out2.branch_distances)
        self.assertEqual(out1.branch_potentials, out2.branch_potentials)
        self.assertEqual(out1.recombined_probabilities, out2.recombined_probabilities)
        self.assertTrue(isclose(sum(out1.recombined_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9))
        self.assertGreaterEqual(out1.concurrence, 0.0)
        self.assertLessEqual(out1.concurrence, 1.0)
        self.assertEqual(len(out1.branch_distances), 4)
        self.assertEqual(len(out1.branch_potentials), 4)
        self.assertEqual(len(out1.branch_forces), 4)
        self.assertEqual(len(out1.branch_phases), 4)
        self.assertEqual(len(out1.recombined_probabilities), 4)

    def test_make_dataset_is_repeatable_and_has_expected_shapes(self):
        ds1 = simulation.make_dataset(num_samples=8, seed=7)
        ds2 = simulation.make_dataset(num_samples=8, seed=7)

        self.assertEqual(len(ds1["inputs"]), 8)
        self.assertEqual(len(ds1["inputs"][0]), 6)
        self.assertEqual(len(ds1["branch_distances"]), 8)
        self.assertEqual(len(ds1["branch_distances"][0]), 4)
        self.assertEqual(len(ds1["branch_potentials"]), 8)
        self.assertEqual(len(ds1["branch_potentials"][0]), 4)
        self.assertEqual(len(ds1["branch_forces"]), 8)
        self.assertEqual(len(ds1["branch_forces"][0]), 4)
        self.assertEqual(len(ds1["gravity_targets"]), 8)
        self.assertEqual(len(ds1["gravity_targets"][0]), 2)
        self.assertEqual(len(ds1["quantum_targets"]), 8)
        self.assertEqual(len(ds1["quantum_targets"][0]), 5)
        self.assertEqual(len(ds1["samples"]), 8)

        self.assertEqual(ds1["inputs"], ds2["inputs"])
        self.assertEqual(ds1["gravity_targets"], ds2["gravity_targets"])
        self.assertEqual(ds1["quantum_targets"], ds2["quantum_targets"])


if __name__ == "__main__":
    unittest.main()
