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
            wavepacket_width=0.3,
            coherence_length=1.4,
        )

        out1 = simulation.oracle(sample)
        out2 = simulation.oracle(sample)

        self.assertEqual(out1.branch_distances, out2.branch_distances)
        self.assertEqual(out1.branch_effective_distances, out2.branch_effective_distances)
        self.assertEqual(out1.branch_potentials, out2.branch_potentials)
        self.assertEqual(out1.recombined_probabilities, out2.recombined_probabilities)
        self.assertTrue(isclose(sum(out1.recombined_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9))
        self.assertGreaterEqual(out1.concurrence, 0.0)
        self.assertLessEqual(out1.concurrence, 1.0)
        self.assertGreaterEqual(out1.visibility, 0.0)
        self.assertLessEqual(out1.visibility, 1.0)
        self.assertEqual(len(out1.branch_distances), 4)
        self.assertEqual(len(out1.branch_effective_distances), 4)
        self.assertEqual(len(out1.branch_potentials), 4)
        self.assertEqual(len(out1.branch_forces), 4)
        self.assertEqual(len(out1.branch_phases), 4)
        self.assertEqual(len(out1.recombined_probabilities), 4)
        self.assertEqual(len(out1.branch_redshift_factors), 4)
        self.assertGreater(out1.branch_effective_distances[0], out1.branch_distances[0])

    def test_make_dataset_is_repeatable_and_has_expected_shapes(self):
        ds1 = simulation.make_dataset(num_samples=8, seed=7, regime="train")
        ds2 = simulation.make_dataset(num_samples=8, seed=7, regime="train")

        self.assertEqual(len(ds1["inputs"]), 8)
        self.assertEqual(len(ds1["inputs"][0]), 8)
        self.assertEqual(len(ds1["branch_distances"]), 8)
        self.assertEqual(len(ds1["branch_distances"][0]), 4)
        self.assertEqual(len(ds1["branch_effective_distances"]), 8)
        self.assertEqual(len(ds1["branch_effective_distances"][0]), 4)
        self.assertEqual(len(ds1["branch_potentials"]), 8)
        self.assertEqual(len(ds1["branch_potentials"][0]), 4)
        self.assertEqual(len(ds1["branch_forces"]), 8)
        self.assertEqual(len(ds1["branch_forces"][0]), 4)
        self.assertEqual(len(ds1["gravity_targets"]), 8)
        self.assertEqual(len(ds1["gravity_targets"][0]), 3)
        self.assertEqual(len(ds1["quantum_targets"]), 8)
        self.assertEqual(len(ds1["quantum_targets"][0]), 6)
        self.assertEqual(len(ds1["samples"]), 8)

        self.assertEqual(ds1["inputs"], ds2["inputs"])
        self.assertEqual(ds1["gravity_targets"], ds2["gravity_targets"])
        self.assertEqual(ds1["quantum_targets"], ds2["quantum_targets"])
        self.assertEqual(ds1["regime"], "train")

    def test_visibility_improves_with_longer_coherence_length(self):
        low = simulation.oracle(
            simulation.OracleSample(
                m1=1.1,
                m2=0.8,
                base_distance=2.7,
                delta1=0.7,
                delta2=0.9,
                interaction_time=1.3,
                wavepacket_width=0.35,
                coherence_length=0.6,
            )
        )
        high = simulation.oracle(
            simulation.OracleSample(
                m1=1.1,
                m2=0.8,
                base_distance=2.7,
                delta1=0.7,
                delta2=0.9,
                interaction_time=1.3,
                wavepacket_width=0.35,
                coherence_length=2.0,
            )
        )

        self.assertLess(low.visibility, high.visibility)

    def test_heldout_regimes_shift_the_input_distribution(self):
        train_ds = simulation.make_dataset(num_samples=32, seed=13, regime="train")
        compact_ds = simulation.make_dataset(num_samples=32, seed=13, regime="heldout_compact")
        decoherent_ds = simulation.make_dataset(num_samples=32, seed=13, regime="heldout_decoherent")

        train_base = sum(row[2] for row in train_ds["inputs"]) / len(train_ds["inputs"])
        compact_base = sum(row[2] for row in compact_ds["inputs"]) / len(compact_ds["inputs"])
        train_coherence = sum(row[7] for row in train_ds["inputs"]) / len(train_ds["inputs"])
        decoherent_coherence = sum(row[7] for row in decoherent_ds["inputs"]) / len(decoherent_ds["inputs"])

        self.assertEqual(compact_ds["regime"], "heldout_compact")
        self.assertEqual(decoherent_ds["regime"], "heldout_decoherent")
        self.assertLess(compact_base, train_base)
        self.assertLess(decoherent_coherence, train_coherence)


if __name__ == "__main__":
    unittest.main()
