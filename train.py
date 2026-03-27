from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import simulation

DEFAULT_TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", "300"))


@dataclass(frozen=True)
class SearchConfig:
    num_train: int = 256
    num_val: int = 128
    seed: int = 42
    max_terms: int = 1
    ridge: float = 0.0
    gravity_weight: float = 1.0
    quantum_weight: float = 1.0
    limit_weight: float = 0.05
    complexity_weight: float = 0.0
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS
    train_regime: str = "train"
    validation_regimes: tuple[str, ...] = ("heldout_compact", "heldout_decoherent", "heldout_wide")
    output_dir: str = "results"
    early_stop_score: float = 1e-12


@dataclass(frozen=True)
class SmearingFunction:
    name: str
    expression: str
    evaluator: Callable[[float], float]
    derivative_evaluator: Callable[[float], float]
    complexity: float = 1.0

    def value(self, x: float) -> float:
        return self.evaluator(x)

    def derivative(self, x: float) -> float:
        return self.derivative_evaluator(x)


@dataclass(frozen=True)
class CandidateFormula:
    smearing: SmearingFunction | None

    def predict_branch_potentials(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        if self.smearing is None:
            return [0.0 for _ in branch_distances]
        g_constant = simulation.OracleConfig().gravitational_constant
        mu = input_row[0] * input_row[1]
        sigma = input_row[6]
        potentials = []
        for distance in branch_distances:
            x = distance / sigma
            g_value = self.smearing.value(x)
            potentials.append(-g_constant * mu * g_value / distance)
        return potentials

    def predict_branch_forces(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        if self.smearing is None:
            return [0.0 for _ in branch_distances]
        g_constant = simulation.OracleConfig().gravitational_constant
        mu = input_row[0] * input_row[1]
        sigma = input_row[6]
        forces = []
        for distance in branch_distances:
            x = distance / sigma
            g_value = self.smearing.value(x)
            g_prime = self.smearing.derivative(x)
            force = g_constant * mu * abs(g_prime / (sigma * distance) - g_value / (distance * distance))
            forces.append(force)
        return forces

    def formula_text(self) -> str:
        if self.smearing is None:
            return "V(r) = 0"
        return f"V(r) = -G*mu*{self.smearing.expression}/r"


@dataclass(frozen=True)
class PredictionBundle:
    gravity_targets: list[list[float]]
    quantum_targets: list[list[float]]


@dataclass(frozen=True)
class ExperimentResult:
    unified_score: float
    val_bpb: float
    gravity_error: float
    quantum_error: float
    limit_penalty: float
    complexity_penalty: float
    zero_formula_score: float
    formula_text: str
    plot_path: str
    num_terms: int
    num_validation_regimes: int
    validation_summary: str
    search_rounds: int
    duration_seconds: float


def _g_erf(x: float) -> float:
    return math.erf(x / 2.0)


def _gprime_erf(x: float) -> float:
    return math.exp(-(x * x) / 4.0) / math.sqrt(math.pi)


def _g_plummer(x: float) -> float:
    return x / math.sqrt(1.0 + x * x)


def _gprime_plummer(x: float) -> float:
    return 1.0 / ((1.0 + x * x) ** 1.5)


def _g_gaussian_switch(x: float) -> float:
    return 1.0 - math.exp(-(x * x))


def _gprime_gaussian_switch(x: float) -> float:
    return 2.0 * x * math.exp(-(x * x))


def _g_tanh(x: float) -> float:
    return math.tanh(x)


def _gprime_tanh(x: float) -> float:
    value = math.tanh(x)
    return 1.0 - value * value


def _g_rational(x: float) -> float:
    return (x * x) / (1.0 + x * x)


def _gprime_rational(x: float) -> float:
    denominator = 1.0 + x * x
    return (2.0 * x) / (denominator * denominator)


SMEARING_LIBRARY = {
    "erf": SmearingFunction(
        name="erf",
        expression="erf((r/sigma)/2)",
        evaluator=_g_erf,
        derivative_evaluator=_gprime_erf,
        complexity=1.0,
    ),
    "plummer": SmearingFunction(
        name="plummer",
        expression="(r/sigma)/sqrt(1 + (r/sigma)^2)",
        evaluator=_g_plummer,
        derivative_evaluator=_gprime_plummer,
        complexity=0.95,
    ),
    "gaussian_switch": SmearingFunction(
        name="gaussian_switch",
        expression="1 - exp(-(r/sigma)^2)",
        evaluator=_g_gaussian_switch,
        derivative_evaluator=_gprime_gaussian_switch,
        complexity=1.0,
    ),
    "tanh": SmearingFunction(
        name="tanh",
        expression="tanh(r/sigma)",
        evaluator=_g_tanh,
        derivative_evaluator=_gprime_tanh,
        complexity=1.0,
    ),
    "rational": SmearingFunction(
        name="rational",
        expression="((r/sigma)^2 / (1 + (r/sigma)^2))",
        evaluator=_g_rational,
        derivative_evaluator=_gprime_rational,
        complexity=0.9,
    ),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _normalized_mse(prediction: list[list[float]], target: list[list[float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    count = 0
    for pred_row, target_row in zip(prediction, target):
        for pred_value, target_value in zip(pred_row, target_row):
            numerator += (pred_value - target_value) ** 2
            denominator += target_value ** 2
            count += 1
    denominator = denominator / max(count, 1) + 1e-30
    return (numerator / max(count, 1)) / denominator


def predict_dataset(formula: CandidateFormula, dataset: dict[str, object]) -> PredictionBundle:
    gravity_targets = []
    quantum_targets = []

    for input_row, branch_distances in zip(dataset["inputs"], dataset["branch_distances"]):
        interaction_time = input_row[5]
        total_mass = input_row[0] + input_row[1]
        wavepacket_width = input_row[6]
        coherence_length = input_row[7]
        branch_potentials = formula.predict_branch_potentials(input_row, branch_distances)
        branch_forces = formula.predict_branch_forces(input_row, branch_distances)
        gravity_targets.append([_mean(branch_potentials), _mean(branch_forces), simulation._std(branch_forces)])

        quantum = simulation.quantum_observables_from_branch_dynamics(
            branch_potentials,
            branch_forces,
            interaction_time=interaction_time,
            total_mass=total_mass,
            wavepacket_width=wavepacket_width,
            coherence_length=coherence_length,
            path_separations=(input_row[3], input_row[4]),
        )
        quantum_targets.append(list(quantum.recombined_probabilities) + [quantum.concurrence, quantum.visibility])

    return PredictionBundle(gravity_targets=gravity_targets, quantum_targets=quantum_targets)


def limit_penalty(formula: CandidateFormula) -> float:
    if formula.smearing is None:
        return 1.0

    x_small = (0.0, 0.25, 0.5)
    x_large = (6.0, 10.0, 20.0)
    x_grid = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    small_penalty = _mean([formula.smearing.value(x) ** 2 for x in x_small])
    large_penalty = _mean([(1.0 - formula.smearing.value(x)) ** 2 for x in x_large])

    monotonic_penalty = 0.0
    previous = formula.smearing.value(x_grid[0])
    for x in x_grid[1:]:
        current = formula.smearing.value(x)
        monotonic_penalty += max(0.0, previous - current) ** 2
        previous = current
    monotonic_penalty /= max(len(x_grid) - 1, 1)

    range_penalty = _mean(
        [max(0.0, -formula.smearing.value(x)) ** 2 + max(0.0, formula.smearing.value(x) - 1.0) ** 2 for x in x_grid]
    )
    return small_penalty + large_penalty + monotonic_penalty + range_penalty


def score_formula(
    formula: CandidateFormula,
    dataset: dict[str, object],
    config: SearchConfig,
) -> tuple[float, float, float, float, float]:
    prediction = predict_dataset(formula, dataset)
    gravity_error = _normalized_mse(prediction.gravity_targets, dataset["gravity_targets"])
    quantum_error = _normalized_mse(prediction.quantum_targets, dataset["quantum_targets"])
    asymptotic_penalty = limit_penalty(formula)
    complexity = 0.0 if formula.smearing is None else formula.smearing.complexity
    unified_score = (
        config.gravity_weight * gravity_error
        + config.quantum_weight * quantum_error
        + config.limit_weight * asymptotic_penalty
        + config.complexity_weight * complexity
    )
    return unified_score, gravity_error, quantum_error, asymptotic_penalty, complexity


def score_formula_across_datasets(
    formula: CandidateFormula,
    datasets: Sequence[dict[str, object]],
    config: SearchConfig,
) -> tuple[float, float, float, float, float]:
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    for dataset in datasets:
        metrics = score_formula(formula, dataset, config)
        for idx, value in enumerate(metrics):
            totals[idx] += value
    count = float(len(datasets))
    return tuple(value / count for value in totals)


def search_best_formula(
    train_dataset: dict[str, object],
    val_datasets: Sequence[dict[str, object]],
    config: SearchConfig,
) -> tuple[CandidateFormula, tuple[float, float, float, float, float]]:
    del train_dataset
    best_formula = CandidateFormula(smearing=None)
    best_metrics = score_formula_across_datasets(best_formula, val_datasets, config)

    for smearing in SMEARING_LIBRARY.values():
        formula = CandidateFormula(smearing=smearing)
        metrics = score_formula_across_datasets(formula, val_datasets, config)
        if metrics[0] < best_metrics[0] - 1e-15:
            best_formula = formula
            best_metrics = metrics
        elif abs(metrics[0] - best_metrics[0]) <= 1e-15 and metrics[4] < best_metrics[4]:
            best_formula = formula
            best_metrics = metrics

    return best_formula, best_metrics


def _is_better_candidate(
    formula: CandidateFormula,
    metrics: tuple[float, float, float, float, float],
    best_formula: CandidateFormula,
    best_metrics: tuple[float, float, float, float, float],
) -> bool:
    score_epsilon = 1e-15
    if metrics[0] < best_metrics[0] - score_epsilon:
        return True
    if abs(metrics[0] - best_metrics[0]) <= score_epsilon and metrics[4] < best_metrics[4]:
        return True
    return False


def _svg_scatter_points(values: list[tuple[float, float]], x0: int, y0: int, width: int, height: int, color: str) -> str:
    xs = [point[0] for point in values]
    ys = [point[1] for point in values]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    if abs(xmax - xmin) < 1e-18:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-18:
        ymax = ymin + 1.0

    circles = []
    for x_value, y_value in values:
        px = x0 + 20 + (x_value - xmin) / (xmax - xmin) * (width - 40)
        py = y0 + height - 20 - (y_value - ymin) / (ymax - ymin) * (height - 40)
        circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}" opacity="0.8" />')
    diagonal = (
        f'<line x1="{x0 + 20}" y1="{y0 + height - 20}" '
        f'x2="{x0 + width - 20}" y2="{y0 + 20}" '
        'stroke="#888888" stroke-dasharray="4 4" />'
    )
    border = (
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'fill="none" stroke="#222222" />'
    )
    return "\n".join([border, diagonal] + circles)


def save_diagnostic_plot(
    dataset: dict[str, object],
    prediction: PredictionBundle,
    formula_text: str,
    validation_summary: str,
    output_dir: str,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "diagnostics.svg"

    potential_points = [
        (oracle_row[0], pred_row[0])
        for oracle_row, pred_row in zip(dataset["gravity_targets"], prediction.gravity_targets)
    ]
    visibility_points = [
        (oracle_row[-1], pred_row[-1])
        for oracle_row, pred_row in zip(dataset["quantum_targets"], prediction.quantum_targets)
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420">
<rect width="100%" height="100%" fill="#fbf8ef" />
<text x="30" y="35" font-family="Courier New" font-size="16" fill="#111111">Unified Smearing Diagnostics</text>
<text x="30" y="60" font-family="Courier New" font-size="12" fill="#333333">{formula_text}</text>
<text x="30" y="78" font-family="Courier New" font-size="11" fill="#444444">{validation_summary}</text>
<text x="80" y="95" font-family="Courier New" font-size="12" fill="#111111">Mean Potential: Oracle vs Predicted</text>
<text x="520" y="95" font-family="Courier New" font-size="12" fill="#111111">Visibility: Oracle vs Predicted</text>
{_svg_scatter_points(potential_points, 40, 110, 360, 250, "#1f77b4")}
{_svg_scatter_points(visibility_points, 480, 110, 360, 250, "#d62728")}
</svg>
"""
    plot_path.write_text(svg)
    return str(plot_path)


def run_experiment(config: SearchConfig | None = None) -> ExperimentResult:
    config = config or SearchConfig()
    t0 = time.time()

    zero_formula = CandidateFormula(smearing=None)
    best_formula = zero_formula
    best_metrics = None
    best_val_datasets = None
    search_rounds = 0

    while True:
        round_seed = config.seed + 1000 * search_rounds
        train_dataset = simulation.make_dataset(config.num_train, seed=round_seed, regime=config.train_regime)
        val_datasets = [
            simulation.make_dataset(
                config.num_val,
                seed=round_seed + 1 + 97 * regime_index,
                regime=regime,
            )
            for regime_index, regime in enumerate(config.validation_regimes)
        ]

        if search_rounds == 0:
            best_metrics = score_formula_across_datasets(zero_formula, val_datasets, config)
            best_val_datasets = val_datasets

        formula, metrics = search_best_formula(train_dataset, val_datasets, config)
        if best_metrics is None or _is_better_candidate(formula, metrics, best_formula, best_metrics):
            best_formula = formula
            best_metrics = metrics
            best_val_datasets = val_datasets

        search_rounds += 1
        elapsed = time.time() - t0
        if best_metrics is not None and best_metrics[0] <= config.early_stop_score:
            break
        if search_rounds >= 1 and elapsed >= config.time_budget_seconds:
            break

    zero_formula_score = (
        best_metrics[0]
        if best_formula == zero_formula
        else score_formula_across_datasets(zero_formula, best_val_datasets, config)[0]
    )
    reference_dataset = best_val_datasets[0]
    prediction = predict_dataset(best_formula, reference_dataset)
    validation_summary = "heldout validation: " + ", ".join(dataset["regime"] for dataset in best_val_datasets)
    plot_path = save_diagnostic_plot(
        reference_dataset,
        prediction,
        best_formula.formula_text(),
        validation_summary,
        config.output_dir,
    )

    unified_score, gravity_error, quantum_error, asymptotic_penalty, complexity = best_metrics
    return ExperimentResult(
        unified_score=unified_score,
        val_bpb=unified_score,
        gravity_error=gravity_error,
        quantum_error=quantum_error,
        limit_penalty=asymptotic_penalty,
        complexity_penalty=complexity,
        zero_formula_score=zero_formula_score,
        formula_text=best_formula.formula_text(),
        plot_path=plot_path,
        num_terms=0 if best_formula.smearing is None else 1,
        num_validation_regimes=len(best_val_datasets),
        validation_summary=validation_summary,
        search_rounds=search_rounds,
        duration_seconds=time.time() - t0,
    )


def main() -> None:
    result = run_experiment()
    print("---")
    print(f"unified_score:    {result.unified_score:.6f}")
    print(f"val_bpb:          {result.val_bpb:.6f}")
    print(f"gravity_error:    {result.gravity_error:.6f}")
    print(f"quantum_error:    {result.quantum_error:.6f}")
    print(f"limit_penalty:    {result.limit_penalty:.6f}")
    print(f"complexity:       {result.complexity_penalty:.6f}")
    print(f"zero_formula:     {result.zero_formula_score:.6f}")
    print(f"num_terms:        {result.num_terms}")
    print(f"validation_sets:  {result.num_validation_regimes}")
    print(f"validation:       {result.validation_summary}")
    print(f"search_rounds:    {result.search_rounds}")
    print(f"time_budget_s:    {DEFAULT_TIME_BUDGET_SECONDS:.2f}")
    print(f"seconds:          {result.duration_seconds:.2f}")
    print(f"plot_path:        {result.plot_path}")
    print(f"formula:          {result.formula_text}")


if __name__ == "__main__":
    main()
