import tempfile
import unittest
from math import pi
from pathlib import Path

import train


class TrainExperimentTests(unittest.TestCase):
    def test_correction_recovery_recovers_known_eft_coefficients_in_amplified_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=160,
                num_val=96,
                seed=31,
                time_budget_seconds=0.01,
                train_regime="eft_sensitive",
                validation_regimes=("eft_sensitive_compact", "eft_sensitive_wide"),
                oracle_config=train.AMPLIFIED_EFT_CONFIG,
                output_dir=tmpdir,
            )
            result = train.run_correction_recovery_experiment(config)

            self.assertLess(result.unified_score, 1e-3)
            self.assertAlmostEqual(result.coefficients["pn_classical"], 3.0, delta=0.15)
            self.assertAlmostEqual(result.coefficients["quantum_eft"], 41.0 / (10.0 * pi), delta=0.15)
            self.assertIn("G*M/(r*c^2)", result.formula_text)
            self.assertIn("G*hbar/(r^2*c^3)", result.formula_text)

    def test_correction_recovery_suppresses_spurious_terms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=160,
                num_val=96,
                seed=37,
                time_budget_seconds=0.01,
                train_regime="eft_sensitive",
                validation_regimes=("eft_sensitive_compact", "eft_sensitive_wide"),
                oracle_config=train.AMPLIFIED_EFT_CONFIG,
                output_dir=tmpdir,
            )
            result = train.run_correction_recovery_experiment(config)

            self.assertAlmostEqual(result.coefficients["smearing_pn_cross"], 0.0, delta=0.1)
            self.assertAlmostEqual(result.coefficients["sigma_over_r"], 0.0, delta=0.1)
            self.assertAlmostEqual(result.coefficients["second_order_pn"], 0.0, delta=0.1)

    def test_uniqueness_sweep_prefers_scaled_erf_and_recovers_half_scale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=48,
                num_val=24,
                seed=17,
                time_budget_seconds=0.01,
                output_dir=tmpdir,
            )
            sweep = train.run_uniqueness_sweep(config)
            best = min(sweep, key=lambda row: row.unified_score)

            self.assertEqual([row.family_name for row in sweep], [
                "erf_scaled",
                "rational_power",
                "stretched_exponential",
                "monotone_spline",
            ])
            self.assertEqual(best.family_name, "erf_scaled")
            self.assertAlmostEqual(best.parameters[0], 0.5, delta=0.05)

            second = sorted(sweep, key=lambda row: row.unified_score)[1]
            self.assertGreater(second.unified_score - best.unified_score, 1e-3)

    def test_run_experiment_respects_small_time_budget_and_records_rounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=24,
                num_val=12,
                seed=5,
                max_terms=2,
                time_budget_seconds=0.01,
                output_dir=tmpdir,
            )
            result = train.run_experiment(config)

            self.assertGreaterEqual(result.search_rounds, 1)
            self.assertGreaterEqual(result.duration_seconds, 0.0)
            self.assertLess(result.duration_seconds, 1.0)

    def test_run_experiment_improves_on_zero_formula(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=48,
                num_val=24,
                seed=11,
                max_terms=2,
                time_budget_seconds=0.01,
                output_dir=tmpdir,
            )
            result = train.run_experiment(config)

            self.assertLess(result.unified_score, result.zero_formula_score)
            self.assertGreaterEqual(result.unified_score, 0.0)
            self.assertAlmostEqual(result.unified_score, result.val_bpb, places=12)
            self.assertGreaterEqual(result.gravity_error, 0.0)
            self.assertGreaterEqual(result.quantum_error, 0.0)
            self.assertTrue(result.formula_text)
            self.assertIn("V(r) = -G*mu*", result.formula_text)

    def test_run_experiment_validates_on_heldout_regimes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=32,
                num_val=16,
                seed=19,
                max_terms=2,
                time_budget_seconds=0.01,
                output_dir=tmpdir,
            )
            result = train.run_experiment(config)

            self.assertIn("heldout", result.validation_summary.lower())
            self.assertGreaterEqual(result.num_validation_regimes, 2)
            self.assertIn("r/sigma", result.formula_text)

    def test_spline_family_is_available_in_uniqueness_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=32,
                num_val=16,
                seed=23,
                time_budget_seconds=0.01,
                output_dir=tmpdir,
            )
            sweep = train.run_uniqueness_sweep(config)
            family_names = [row.family_name for row in sweep]

            self.assertIn("monotone_spline", family_names)

    def test_run_experiment_writes_a_nonempty_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = train.SearchConfig(
                num_train=32,
                num_val=16,
                seed=3,
                max_terms=2,
                time_budget_seconds=0.01,
                output_dir=tmpdir,
            )
            result = train.run_experiment(config)
            plot_path = Path(result.plot_path)

            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
