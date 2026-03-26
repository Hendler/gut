from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path

import simulation

DEFAULT_TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", "300"))


@dataclass(frozen=True)
class SearchConfig:
    num_train: int = 256
    num_val: int = 128
    seed: int = 42
    max_terms: int = 3
    ridge: float = 1e-8
    gravity_weight: float = 1.0
    quantum_weight: float = 1.0
    limit_weight: float = 0.1
    complexity_weight: float = 0.0
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS
    output_dir: str = "results"


@dataclass(frozen=True)
class FormulaTerm:
    name: str

    def value(self, input_row: list[float], r: float) -> float:
        mu = input_row[0] * input_row[1]
        total_mass = input_row[0] + input_row[1]
        wavepacket_width = input_row[6]
        r_eff = simulation.effective_distance(r, wavepacket_width)
        if self.name == "mu/r":
            return mu / r
        if self.name == "mu/r_eff":
            return mu / r_eff
        if self.name == "mu/r_eff^2":
            return mu / (r_eff * r_eff)
        if self.name == "mu/r_eff^3":
            return mu / (r_eff * r_eff * r_eff)
        if self.name == "mu*M/r_eff^2":
            return mu * total_mass / (r_eff * r_eff)
        if self.name == "mu*sigma^2/r_eff^3":
            return mu * wavepacket_width * wavepacket_width / (r_eff * r_eff * r_eff)
        raise KeyError(f"Unknown basis term: {self.name}")

    def derivative(self, input_row: list[float], r: float) -> float:
        mu = input_row[0] * input_row[1]
        total_mass = input_row[0] + input_row[1]
        wavepacket_width = input_row[6]
        r_eff = simulation.effective_distance(r, wavepacket_width)
        if self.name == "mu/r":
            return -mu / (r * r)
        if self.name == "mu/r_eff":
            return -mu * r / (r_eff ** 3)
        if self.name == "mu/r_eff^2":
            return -2.0 * mu * r / (r_eff ** 4)
        if self.name == "mu/r_eff^3":
            return -3.0 * mu * r / (r_eff ** 5)
        if self.name == "mu*M/r_eff^2":
            return -2.0 * mu * total_mass * r / (r_eff ** 4)
        if self.name == "mu*sigma^2/r_eff^3":
            return -3.0 * mu * wavepacket_width * wavepacket_width * r / (r_eff ** 5)
        raise KeyError(f"Unknown basis term: {self.name}")


@dataclass(frozen=True)
class CandidateFormula:
    terms: tuple[FormulaTerm, ...]
    coefficients: tuple[float, ...]

    def predict_branch_potentials(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        if not self.terms:
            return [0.0 for _ in branch_distances]
        potentials = []
        for distance in branch_distances:
            value = 0.0
            for coefficient, term in zip(self.coefficients, self.terms):
                value += coefficient * term.value(input_row, distance)
            potentials.append(value)
        return potentials

    def predict_branch_forces(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        if not self.terms:
            return [0.0 for _ in branch_distances]
        forces = []
        for distance in branch_distances:
            derivative = 0.0
            for coefficient, term in zip(self.coefficients, self.terms):
                derivative += coefficient * term.derivative(input_row, distance)
            forces.append(abs(-derivative))
        return forces

    def formula_text(self) -> str:
        if not self.terms:
            return "V(r) = 0"
        pieces = []
        for coefficient, term in zip(self.coefficients, self.terms):
            pieces.append(f"{coefficient:+.6f}*{term.name}")
        return "V(r) = " + " ".join(pieces)


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
    search_rounds: int
    duration_seconds: float


BASIS_LIBRARY = tuple(
    FormulaTerm(name)
    for name in ("mu/r", "mu/r_eff", "mu/r_eff^2", "mu/r_eff^3", "mu*M/r_eff^2", "mu*sigma^2/r_eff^3")
)


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
    denominator = denominator / max(count, 1) + 1e-12
    return (numerator / max(count, 1)) / denominator


def _design_matrix(dataset: dict[str, object], terms: tuple[FormulaTerm, ...]) -> list[list[float]]:
    matrix = []
    inputs = dataset["inputs"]
    distances = dataset["branch_distances"]
    for input_row, distance_row in zip(inputs, distances):
        for distance in distance_row:
            matrix.append([term.value(input_row, distance) for term in terms])
    return matrix


def _flatten_branch_targets(dataset: dict[str, object]) -> list[float]:
    return [value for row in dataset["branch_potentials"] for value in row]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for pivot in range(size):
        max_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        augmented[pivot], augmented[max_row] = augmented[max_row], augmented[pivot]

        pivot_value = augmented[pivot][pivot]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Singular matrix encountered while fitting formula coefficients.")

        scale = 1.0 / pivot_value
        for column in range(pivot, size + 1):
            augmented[pivot][column] *= scale

        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            for column in range(pivot, size + 1):
                augmented[row][column] -= factor * augmented[pivot][column]

    return [augmented[row][-1] for row in range(size)]


def fit_formula(
    train_dataset: dict[str, object],
    terms: tuple[FormulaTerm, ...],
    ridge: float,
) -> CandidateFormula:
    if not terms:
        return CandidateFormula(terms=(), coefficients=())

    design = _design_matrix(train_dataset, terms)
    target = _flatten_branch_targets(train_dataset)
    size = len(terms)

    gram = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    for row, y_value in zip(design, target):
        for i in range(size):
            rhs[i] += row[i] * y_value
            for j in range(size):
                gram[i][j] += row[i] * row[j]

    for i in range(size):
        gram[i][i] += ridge

    coefficients = tuple(_solve_linear_system(gram, rhs))
    return CandidateFormula(terms=terms, coefficients=coefficients)


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
        )
        quantum_targets.append(list(quantum.recombined_probabilities) + [quantum.concurrence, quantum.visibility])

    return PredictionBundle(
        gravity_targets=gravity_targets,
        quantum_targets=quantum_targets,
    )


def limit_penalty(formula: CandidateFormula) -> float:
    if not formula.terms:
        return 1.0

    config = simulation.OracleConfig()
    pn_prefactor = (
        config.gravitational_constant * config.gravitational_constant
        / (2.0 * config.speed_of_light * config.speed_of_light)
    )
    prediction = []
    target = []
    for m1, m2, distance, sigma in ((1.0, 0.8, 8.0, 0.2), (1.2, 1.0, 12.0, 0.3), (1.4, 1.1, 20.0, 0.4)):
        pred_row = []
        target_row = []
        for offset in (0.0, 2.0, 6.0):
            mu = m1 * m2
            total_mass = m1 + m2
            shifted_distance = distance + offset
            input_row = [m1, m2, shifted_distance, 0.2, 0.2, 1.0, sigma, 1.5]
            r_eff = simulation.effective_distance(shifted_distance, sigma)
            pred_row.append(formula.predict_branch_potentials(input_row, [shifted_distance])[0])
            target_row.append(-mu / r_eff - pn_prefactor * mu * total_mass / (r_eff * r_eff))
        prediction.append(pred_row)
        target.append(target_row)
    return _normalized_mse(prediction, target)


def score_formula(
    formula: CandidateFormula,
    dataset: dict[str, object],
    config: SearchConfig,
) -> tuple[float, float, float, float, float]:
    prediction = predict_dataset(formula, dataset)
    gravity_error = _normalized_mse(prediction.gravity_targets, dataset["gravity_targets"])
    quantum_error = _normalized_mse(prediction.quantum_targets, dataset["quantum_targets"])
    asymptotic_penalty = limit_penalty(formula)
    complexity = float(len(formula.terms))
    unified_score = (
        config.gravity_weight * gravity_error
        + config.quantum_weight * quantum_error
        + config.limit_weight * asymptotic_penalty
        + config.complexity_weight * complexity
    )
    return unified_score, gravity_error, quantum_error, asymptotic_penalty, complexity


def search_best_formula(
    train_dataset: dict[str, object],
    val_dataset: dict[str, object],
    config: SearchConfig,
) -> tuple[CandidateFormula, tuple[float, float, float, float, float]]:
    best_formula = CandidateFormula(terms=(), coefficients=())
    best_metrics = score_formula(best_formula, val_dataset, config)

    max_terms = min(config.max_terms, len(BASIS_LIBRARY))
    for num_terms in range(1, max_terms + 1):
        for subset in itertools.combinations(BASIS_LIBRARY, num_terms):
            formula = fit_formula(train_dataset, subset, ridge=config.ridge)
            metrics = score_formula(formula, val_dataset, config)
            if metrics[0] < best_metrics[0]:
                best_formula = formula
                best_metrics = metrics

    return best_formula, best_metrics


def _is_better_candidate(
    formula: CandidateFormula,
    metrics: tuple[float, float, float, float, float],
    best_formula: CandidateFormula,
    best_metrics: tuple[float, float, float, float, float],
) -> bool:
    score_epsilon = 1e-12
    if metrics[0] < best_metrics[0] - score_epsilon:
        return True
    if abs(metrics[0] - best_metrics[0]) <= score_epsilon and len(formula.terms) < len(best_formula.terms):
        return True
    return False


def _svg_scatter_points(values: list[tuple[float, float]], x0: int, y0: int, width: int, height: int, color: str) -> str:
    xs = [point[0] for point in values]
    ys = [point[1] for point in values]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0

    circles = []
    for x_value, y_value in values:
        px = x0 + 20 + (x_value - xmin) / (xmax - xmin) * (width - 40)
        py = y0 + height - 20 - (y_value - ymin) / (ymax - ymin) * (height - 40)
        circles.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}" opacity="0.8" />'
        )
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
<text x="30" y="35" font-family="Courier New" font-size="16" fill="#111111">Unified Formula Diagnostics</text>
<text x="30" y="60" font-family="Courier New" font-size="12" fill="#333333">{formula_text}</text>
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

    zero_formula = CandidateFormula(terms=(), coefficients=())
    best_formula = zero_formula
    best_metrics = None
    best_val_dataset = None
    search_rounds = 0

    while True:
        round_seed = config.seed + 1000 * search_rounds
        train_dataset = simulation.make_dataset(config.num_train, seed=round_seed)
        val_dataset = simulation.make_dataset(config.num_val, seed=round_seed + 1)

        if search_rounds == 0:
            best_metrics = score_formula(zero_formula, val_dataset, config)
            best_val_dataset = val_dataset

        formula, metrics = search_best_formula(train_dataset, val_dataset, config)
        if best_metrics is None or _is_better_candidate(formula, metrics, best_formula, best_metrics):
            best_formula = formula
            best_metrics = metrics
            best_val_dataset = val_dataset

        search_rounds += 1
        if search_rounds >= 1 and (time.time() - t0) >= config.time_budget_seconds:
            break

    zero_formula_score = best_metrics[0] if best_formula == zero_formula else score_formula(zero_formula, best_val_dataset, config)[0]
    prediction = predict_dataset(best_formula, best_val_dataset)
    plot_path = save_diagnostic_plot(
        best_val_dataset,
        prediction,
        best_formula.formula_text(),
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
        num_terms=len(best_formula.terms),
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
    print(f"search_rounds:    {result.search_rounds}")
    print(f"time_budget_s:    {DEFAULT_TIME_BUDGET_SECONDS:.2f}")
    print(f"seconds:          {result.duration_seconds:.2f}")
    print(f"plot_path:        {result.plot_path}")
    print(f"formula:          {result.formula_text}")


if __name__ == "__main__":
    main()
