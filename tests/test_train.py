import tempfile
import unittest
from pathlib import Path

import train


class TrainExperimentTests(unittest.TestCase):
    def test_erf_smearing_beats_plummer_surrogate_on_oracle_data(self):
        dataset = train.simulation.make_dataset(num_samples=24, seed=17, regime="heldout_compact")
        config = train.SearchConfig(num_train=24, num_val=12, time_budget_seconds=0.01)

        erf_formula = train.CandidateFormula(smearing=train.SMEARING_LIBRARY["erf"])
        plummer_formula = train.CandidateFormula(smearing=train.SMEARING_LIBRARY["plummer"])

        erf_score = train.score_formula(erf_formula, dataset, config)[0]
        plummer_score = train.score_formula(plummer_formula, dataset, config)[0]

        self.assertLess(erf_score, plummer_score)

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
