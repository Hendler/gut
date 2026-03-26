import unittest
from math import isclose

import simulation


class SimulationOracleTests(unittest.TestCase):
    def test_gaussian_smeared_potential_recovers_newtonian_limit_far_away(self):
        sample = simulation.OracleSample(
            m1=1.6e-14,
            m2=1.2e-14,
            base_distance=9.0e-4,
            delta1=4.0e-5,
            delta2=5.0e-5,
            interaction_time=1.0,
            wavepacket_width=4.0e-6,
            coherence_length=1.5e-4,
        )
        distance = sample.base_distance
        softened = simulation.branch_potential(sample, distance)
        point_mass = -simulation.OracleConfig().gravitational_constant * sample.m1 * sample.m2 / distance

        self.assertTrue(isclose(softened, point_mass, rel_tol=5e-4, abs_tol=0.0))

    def test_gaussian_smearing_softens_short_range_force(self):
        sample = simulation.OracleSample(
            m1=2.0e-14,
            m2=1.8e-14,
            base_distance=7.0e-5,
            delta1=1.0e-5,
            delta2=1.2e-5,
            interaction_time=1.0,
            wavepacket_width=3.5e-5,
            coherence_length=1.2e-4,
        )
        distance = sample.base_distance
        softened_force = simulation.branch_force(sample, distance)
        point_mass_force = (
            simulation.OracleConfig().gravitational_constant * sample.m1 * sample.m2 / (distance * distance)
        )

        self.assertLess(softened_force, point_mass_force)

    def test_oracle_outputs_are_well_formed_and_deterministic(self):
        sample = simulation.OracleSample(
            m1=1.2e-14,
            m2=0.9e-14,
            base_distance=2.4e-4,
            delta1=4.0e-5,
            delta2=6.0e-5,
            interaction_time=1.1,
            wavepacket_width=1.8e-5,
            coherence_length=1.4e-4,
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
        self.assertGreaterEqual(out1.branch_effective_distances[0], out1.branch_distances[0])

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
                m1=1.1e-14,
                m2=0.8e-14,
                base_distance=2.7e-4,
                delta1=7.0e-5,
                delta2=9.0e-5,
                interaction_time=1.3,
                wavepacket_width=2.2e-5,
                coherence_length=3.0e-5,
            )
        )
        high = simulation.oracle(
            simulation.OracleSample(
                m1=1.1e-14,
                m2=0.8e-14,
                base_distance=2.7e-4,
                delta1=7.0e-5,
                delta2=9.0e-5,
                interaction_time=1.3,
                wavepacket_width=2.2e-5,
                coherence_length=2.0e-4,
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
