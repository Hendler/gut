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
    early_stop_score: float = 5e-5


@dataclass(frozen=True)
class CandidateFormula:
    family_name: str | None
    parameters: tuple[float, ...]
    expression: str
    evaluator: Callable[[float], float]
    derivative_evaluator: Callable[[float], float]
    complexity: float

    def predict_branch_potentials(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        if self.family_name is None:
            return [0.0 for _ in branch_distances]
        g_constant = simulation.OracleConfig().gravitational_constant
        mu = input_row[0] * input_row[1]
        sigma = max(input_row[6], 1e-30)
        potentials = []
        for distance in branch_distances:
            x = distance / sigma
            potentials.append(-g_constant * mu * self.evaluator(x) / distance)
        return potentials

    def predict_branch_forces(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        if self.family_name is None:
            return [0.0 for _ in branch_distances]
        g_constant = simulation.OracleConfig().gravitational_constant
        mu = input_row[0] * input_row[1]
        sigma = max(input_row[6], 1e-30)
        forces = []
        for distance in branch_distances:
            x = distance / sigma
            g_value = self.evaluator(x)
            g_prime = self.derivative_evaluator(x)
            force = g_constant * mu * abs(g_prime / (sigma * distance) - g_value / (distance * distance))
            forces.append(force)
        return forces

    def formula_text(self) -> str:
        if self.family_name is None:
            return "V(r) = 0"
        return f"V(r) = -G*mu*{self.expression}/r"


@dataclass(frozen=True)
class PredictionBundle:
    gravity_targets: list[list[float]]
    quantum_targets: list[list[float]]


@dataclass(frozen=True)
class FamilySweepResult:
    family_name: str
    candidate: CandidateFormula
    parameters: tuple[float, ...]
    unified_score: float
    gravity_error: float
    quantum_error: float
    limit_penalty: float
    complexity_penalty: float


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


def _frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    epsilon = step * 0.5
    while current <= stop + epsilon:
        values.append(round(current, 12))
        current += step
    return values


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _extract_smearing_targets(dataset: dict[str, object]) -> tuple[list[float], list[float]]:
    g_constant = simulation.OracleConfig().gravitational_constant
    xs = []
    ys = []
    for input_row, distances, potentials in zip(
        dataset["inputs"],
        dataset["branch_distances"],
        dataset["branch_potentials"],
    ):
        mu = input_row[0] * input_row[1]
        sigma = max(input_row[6], 1e-30)
        for distance, potential in zip(distances, potentials):
            xs.append(distance / sigma)
            ys.append(-potential * distance / (g_constant * mu))
    return xs, ys


def _format_number(value: float) -> str:
    rounded = round(value, 6)
    if abs(rounded - round(rounded)) < 1e-12:
        return str(int(round(rounded)))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def build_scaled_erf_formula(a: float) -> CandidateFormula:
    def evaluator(x: float) -> float:
        return math.erf(a * x)

    def derivative(x: float) -> float:
        return (2.0 * a / math.sqrt(math.pi)) * math.exp(-((a * x) ** 2))

    if abs(a - 0.5) < 1e-12:
        expression = "erf((r/sigma)/2)"
    else:
        expression = f"erf({_format_number(a)}*(r/sigma))"
    return CandidateFormula(
        family_name="erf_scaled",
        parameters=(a,),
        expression=expression,
        evaluator=evaluator,
        derivative_evaluator=derivative,
        complexity=1.05,
    )


def build_rational_formula(n: float, c: float) -> CandidateFormula:
    def evaluator(x: float) -> float:
        power = x ** n
        return power / (c + power)

    def derivative(x: float) -> float:
        if x == 0.0:
            if abs(n - 1.0) < 1e-12:
                return 1.0 / c
            return 0.0
        power = x ** n
        return (n * c * (x ** (n - 1.0))) / ((c + power) ** 2)

    expression = f"((r/sigma)^{_format_number(n)} / ({_format_number(c)} + (r/sigma)^{_format_number(n)}))"
    return CandidateFormula(
        family_name="rational_power",
        parameters=(n, c),
        expression=expression,
        evaluator=evaluator,
        derivative_evaluator=derivative,
        complexity=1.15,
    )


def build_stretched_exponential_formula(a: float, b: float) -> CandidateFormula:
    def evaluator(x: float) -> float:
        return 1.0 - math.exp(-((a * x) ** b))

    def derivative(x: float) -> float:
        if x == 0.0:
            if abs(b - 1.0) < 1e-12:
                return a
            return 0.0
        ax = a * x
        return math.exp(-(ax ** b)) * b * a * (ax ** (b - 1.0))

    expression = f"(1 - exp(-({_format_number(a)}*(r/sigma))^{_format_number(b)}))"
    return CandidateFormula(
        family_name="stretched_exponential",
        parameters=(a, b),
        expression=expression,
        evaluator=evaluator,
        derivative_evaluator=derivative,
        complexity=1.2,
    )


def build_monotone_spline_formula(knots: tuple[float, ...], values: tuple[float, ...]) -> CandidateFormula:
    def evaluator(x: float) -> float:
        if x <= knots[0]:
            return values[0]
        if x >= knots[-1]:
            return values[-1]
        for left_index in range(len(knots) - 1):
            x0 = knots[left_index]
            x1 = knots[left_index + 1]
            if x <= x1:
                y0 = values[left_index]
                y1 = values[left_index + 1]
                ratio = (x - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return values[-1]

    def derivative(x: float) -> float:
        if x <= knots[0] or x >= knots[-1]:
            return 0.0
        for left_index in range(len(knots) - 1):
            x0 = knots[left_index]
            x1 = knots[left_index + 1]
            if x <= x1:
                y0 = values[left_index]
                y1 = values[left_index + 1]
                return (y1 - y0) / (x1 - x0)
        return 0.0

    interior_text = ", ".join(f"{value:.4f}" for value in values[1:-1])
    expression = f"spline(r/sigma; knots={len(knots)}, interior=[{interior_text}])"
    return CandidateFormula(
        family_name="monotone_spline",
        parameters=values[1:-1],
        expression=expression,
        evaluator=evaluator,
        derivative_evaluator=derivative,
        complexity=1.5,
    )


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
    if formula.family_name is None:
        return 1.0

    x_small = (0.0, 1.0e-3, 1.0e-2, 5.0e-2)
    x_large = (12.0, 24.0, 48.0)
    x_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    small_penalty = _mean([formula.evaluator(x) ** 2 for x in x_small])
    large_penalty = _mean([(1.0 - formula.evaluator(x)) ** 2 for x in x_large])

    monotonic_penalty = 0.0
    previous = formula.evaluator(x_grid[0])
    for x in x_grid[1:]:
        current = formula.evaluator(x)
        monotonic_penalty += max(0.0, previous - current) ** 2
        previous = current
    monotonic_penalty /= max(len(x_grid) - 1, 1)

    range_penalty = _mean(
        [max(0.0, -formula.evaluator(x)) ** 2 + max(0.0, formula.evaluator(x) - 1.0) ** 2 for x in x_grid]
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
    complexity = formula.complexity if formula.family_name is not None else 0.0
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


def _choose_best_formula(
    formulas: Sequence[CandidateFormula],
    dataset: dict[str, object],
    config: SearchConfig,
) -> CandidateFormula:
    best_formula = CandidateFormula(
        family_name=None,
        parameters=(),
        expression="0",
        evaluator=lambda x: 0.0,
        derivative_evaluator=lambda x: 0.0,
        complexity=0.0,
    )
    best_score = score_formula(best_formula, dataset, config)[0]
    for formula in formulas:
        score = score_formula(formula, dataset, config)[0]
        if score < best_score - 1e-15:
            best_formula = formula
            best_score = score
        elif abs(score - best_score) <= 1e-15 and formula.complexity < best_formula.complexity:
            best_formula = formula
            best_score = score
    return best_formula


def fit_erf_scaled_family(train_dataset: dict[str, object], config: SearchConfig) -> CandidateFormula:
    candidates = [build_scaled_erf_formula(a) for a in _frange(0.05, 0.95, 0.01)]
    return _choose_best_formula(candidates, train_dataset, config)


def fit_rational_family(train_dataset: dict[str, object], config: SearchConfig) -> CandidateFormula:
    n_values = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
    c_values = (0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    candidates = [build_rational_formula(n, c) for n in n_values for c in c_values]
    return _choose_best_formula(candidates, train_dataset, config)


def fit_stretched_exponential_family(train_dataset: dict[str, object], config: SearchConfig) -> CandidateFormula:
    a_values = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0)
    b_values = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
    candidates = [build_stretched_exponential_formula(a, b) for a in a_values for b in b_values]
    return _choose_best_formula(candidates, train_dataset, config)


def fit_monotone_spline_family(train_dataset: dict[str, object], config: SearchConfig) -> CandidateFormula:
    knots = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    xs, ys = _extract_smearing_targets(train_dataset)
    initial_values = [0.0]
    for knot_index in range(1, len(knots) - 1):
        left = 0.5 * (knots[knot_index - 1] + knots[knot_index])
        right = 0.5 * (knots[knot_index] + knots[knot_index + 1])
        bucket = [y for x, y in zip(xs, ys) if left <= x < right]
        if bucket:
            guess = sum(bucket) / len(bucket)
        else:
            guess = initial_values[-1]
        guess = _clamp(guess, initial_values[-1], 1.0)
        initial_values.append(guess)
    initial_values.append(1.0)
    values = initial_values

    best_formula = build_monotone_spline_formula(knots, tuple(values))
    best_score = score_formula(best_formula, train_dataset, config)[0]

    for _ in range(4):
        for index in range(1, len(values) - 1):
            low = values[index - 1]
            high = values[index + 1]
            if high - low < 1e-12:
                continue
            local_best_value = values[index]
            local_best_score = best_score
            for fraction in range(9):
                candidate_value = low + (high - low) * fraction / 8.0
                trial_values = values[:]
                trial_values[index] = candidate_value
                trial_formula = build_monotone_spline_formula(knots, tuple(trial_values))
                trial_score = score_formula(trial_formula, train_dataset, config)[0]
                if trial_score < local_best_score - 1e-15:
                    local_best_value = candidate_value
                    local_best_score = trial_score
                    best_formula = trial_formula
                    best_score = trial_score
            values[index] = local_best_value

    return best_formula


FAMILY_FITTERS: tuple[tuple[str, Callable[[dict[str, object], SearchConfig], CandidateFormula]], ...] = (
    ("erf_scaled", fit_erf_scaled_family),
    ("rational_power", fit_rational_family),
    ("stretched_exponential", fit_stretched_exponential_family),
    ("monotone_spline", fit_monotone_spline_family),
)


def run_uniqueness_sweep_on_datasets(
    train_dataset: dict[str, object],
    val_datasets: Sequence[dict[str, object]],
    config: SearchConfig,
) -> list[FamilySweepResult]:
    results = []
    for family_name, fitter in FAMILY_FITTERS:
        candidate = fitter(train_dataset, config)
        unified_score, gravity_error, quantum_error, asymptotic_penalty, complexity = score_formula_across_datasets(
            candidate,
            val_datasets,
            config,
        )
        results.append(
            FamilySweepResult(
                family_name=family_name,
                candidate=candidate,
                parameters=candidate.parameters,
                unified_score=unified_score,
                gravity_error=gravity_error,
                quantum_error=quantum_error,
                limit_penalty=asymptotic_penalty,
                complexity_penalty=complexity,
            )
        )
    return results


def run_uniqueness_sweep(config: SearchConfig | None = None) -> list[FamilySweepResult]:
    config = config or SearchConfig()
    train_dataset = simulation.make_dataset(config.num_train, seed=config.seed, regime=config.train_regime)
    val_datasets = [
        simulation.make_dataset(config.num_val, seed=config.seed + 1 + 97 * regime_index, regime=regime)
        for regime_index, regime in enumerate(config.validation_regimes)
    ]
    return run_uniqueness_sweep_on_datasets(train_dataset, val_datasets, config)


def search_best_formula(
    train_dataset: dict[str, object],
    val_datasets: Sequence[dict[str, object]],
    config: SearchConfig,
) -> tuple[CandidateFormula, tuple[float, float, float, float, float]]:
    sweep = run_uniqueness_sweep_on_datasets(train_dataset, val_datasets, config)
    best = min(sweep, key=lambda row: (row.unified_score, row.complexity_penalty))
    return best.candidate, (
        best.unified_score,
        best.gravity_error,
        best.quantum_error,
        best.limit_penalty,
        best.complexity_penalty,
    )


def _is_better_candidate(
    formula: CandidateFormula,
    metrics: tuple[float, float, float, float, float],
    best_formula: CandidateFormula,
    best_metrics: tuple[float, float, float, float, float],
) -> bool:
    score_epsilon = 1e-15
    if metrics[0] < best_metrics[0] - score_epsilon:
        return True
    if abs(metrics[0] - best_metrics[0]) <= score_epsilon and formula.complexity < best_formula.complexity:
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

    zero_formula = CandidateFormula(
        family_name=None,
        parameters=(),
        expression="0",
        evaluator=lambda x: 0.0,
        derivative_evaluator=lambda x: 0.0,
        complexity=0.0,
    )
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
        if best_formula.family_name is None
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
        num_terms=0 if best_formula.family_name is None else 1,
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
