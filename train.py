from __future__ import annotations

import argparse
import itertools
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import simulation

DEFAULT_TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", "300"))
AMPLIFIED_EFT_CONFIG = simulation.OracleConfig(
    gravitational_constant=1.0,
    hbar=0.5,
    speed_of_light=2.0,
    min_distance=0.05,
)


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
    oracle_config: simulation.OracleConfig = field(default_factory=simulation.OracleConfig)


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


@dataclass(frozen=True)
class CorrectionTerm:
    name: str
    expression: str
    g_power: int
    hbar_power: int
    mass_power: int
    distance_power: int
    sigma_power: int
    c_power: int

    def value(self, input_row: list[float], distance: float, config: simulation.OracleConfig) -> float:
        total_mass = input_row[0] + input_row[1]
        sigma = input_row[6]
        return (
            (config.gravitational_constant ** self.g_power)
            * (config.hbar ** self.hbar_power)
            * (total_mass ** self.mass_power)
            * (distance ** self.distance_power)
            * (sigma ** self.sigma_power)
            * (config.speed_of_light ** self.c_power)
        )

    def derivative(self, input_row: list[float], distance: float, config: simulation.OracleConfig) -> float:
        if self.distance_power == 0:
            return 0.0
        return self.distance_power * self.value(input_row, distance, config) / distance


@dataclass(frozen=True)
class CorrectionFormula:
    terms: tuple[CorrectionTerm, ...]
    coefficients: tuple[float, ...]
    physics_config: simulation.OracleConfig
    complexity: float
    family_name: str = "eft_residual"

    def predict_branch_potentials(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        potentials = []
        g_constant = self.physics_config.gravitational_constant
        mu = input_row[0] * input_row[1]
        sigma = max(input_row[6], 1e-30)
        for distance in branch_distances:
            smearing = math.erf(distance / (2.0 * sigma))
            correction = 1.0
            for coefficient, term in zip(self.coefficients, self.terms):
                correction += coefficient * term.value(input_row, distance, self.physics_config)
            potentials.append(-g_constant * mu * smearing * correction / distance)
        return potentials

    def predict_branch_forces(self, input_row: list[float], branch_distances: list[float]) -> list[float]:
        forces = []
        g_constant = self.physics_config.gravitational_constant
        mu = input_row[0] * input_row[1]
        sigma = max(input_row[6], 1e-30)
        for distance in branch_distances:
            smearing = math.erf(distance / (2.0 * sigma))
            smearing_derivative = math.exp(-(distance * distance) / (4.0 * sigma * sigma)) / (math.sqrt(math.pi) * sigma)
            correction = 1.0
            correction_derivative = 0.0
            for coefficient, term in zip(self.coefficients, self.terms):
                correction += coefficient * term.value(input_row, distance, self.physics_config)
                correction_derivative += coefficient * term.derivative(input_row, distance, self.physics_config)
            total_derivative = (
                smearing_derivative * correction / distance
                + smearing * correction_derivative / distance
                - smearing * correction / (distance * distance)
            )
            forces.append(abs(g_constant * mu * total_derivative))
        return forces

    def formula_text(self) -> str:
        if not self.terms:
            return "V(r) = -G*mu*erf((r/sigma)/2)/r"
        pieces = []
        for coefficient, term in zip(self.coefficients, self.terms):
            pieces.append(f"{coefficient:+.6f}*{term.expression}")
        return "V(r) = -G*mu*erf((r/sigma)/2)/r * [1 " + " ".join(pieces) + "]"

    def coefficient_map(
        self,
        all_terms: Sequence[CorrectionTerm] | None = None,
        *,
        key: str = "name",
    ) -> dict[str, float]:
        all_terms = all_terms or CORRECTION_LIBRARY
        if key not in {"name", "expression"}:
            raise ValueError(f"Unknown coefficient key mode: {key}")
        key_fn = (lambda term: term.name) if key == "name" else (lambda term: term.expression)
        values = {key_fn(term): 0.0 for term in all_terms}
        for coefficient, term in zip(self.coefficients, self.terms):
            values[key_fn(term)] = coefficient
        return values


def canonicalize_correction_formula(
    formula: CorrectionFormula,
    coefficient_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-6,
) -> CorrectionFormula:
    scale = max((abs(coefficient) for coefficient in formula.coefficients), default=0.0)
    effective_tolerance = max(coefficient_tolerance, relative_tolerance * scale)
    kept_terms = []
    kept_coefficients = []
    for term, coefficient in zip(formula.terms, formula.coefficients):
        if abs(coefficient) > effective_tolerance:
            kept_terms.append(term)
            kept_coefficients.append(coefficient)
    return CorrectionFormula(
        terms=tuple(kept_terms),
        coefficients=tuple(kept_coefficients),
        physics_config=formula.physics_config,
        complexity=1.0 + 0.2 * len(kept_terms),
    )


@dataclass(frozen=True)
class CorrectionRecoveryResult:
    unified_score: float
    gravity_error: float
    quantum_error: float
    zero_formula_score: float
    formula_text: str
    plot_path: str
    coefficients: dict[str, float]
    selected_term_names: tuple[str, ...]
    validation_summary: str
    num_validation_regimes: int
    search_rounds: int
    duration_seconds: float


@dataclass(frozen=True)
class CorrectionRankingEntry:
    formula: CorrectionFormula
    unified_score: float
    gravity_error: float
    quantum_error: float
    limit_penalty: float
    complexity_penalty: float
    selected_term_names: tuple[str, ...]
    coefficients: dict[str, float]

    @property
    def formula_text(self) -> str:
        return self.formula.formula_text()


@dataclass(frozen=True)
class BlindDegeneracyReport:
    top_rankings: tuple[CorrectionRankingEntry, ...]
    runner_up_margin: float
    total_subsets: int
    zero_formula_score: float
    plot_path: str
    validation_summary: str
    duration_seconds: float


def _power_factor_text(symbol: str, exponent: int) -> str:
    if exponent == 1:
        return symbol
    return f"{symbol}^{exponent}"


def _format_dimensionless_term_expression(
    g_power: int,
    hbar_power: int,
    mass_power: int,
    distance_power: int,
    sigma_power: int,
    c_power: int,
) -> str:
    numerator = []
    denominator = []
    for symbol, exponent in (
        ("G", g_power),
        ("hbar", hbar_power),
        ("M", mass_power),
        ("sigma", sigma_power),
        ("r", distance_power),
        ("c", c_power),
    ):
        if exponent > 0:
            numerator.append(_power_factor_text(symbol, exponent))
        elif exponent < 0:
            denominator.append(_power_factor_text(symbol, -exponent))

    numerator_text = "*".join(numerator) if numerator else "1"
    denominator_text = "*".join(denominator)
    if not denominator_text:
        return numerator_text
    if numerator_text == "1":
        if "*" in denominator_text:
            return f"1/({denominator_text})"
        return f"1/{denominator_text}"
    if "*" in denominator_text:
        return f"{numerator_text}/({denominator_text})"
    return f"{numerator_text}/{denominator_text}"


def make_correction_term(
    name: str,
    *,
    g_power: int,
    hbar_power: int,
    mass_power: int,
    distance_power: int,
    sigma_power: int,
    c_power: int,
    expression: str | None = None,
) -> CorrectionTerm:
    return CorrectionTerm(
        name=name,
        expression=expression
        or _format_dimensionless_term_expression(
            g_power,
            hbar_power,
            mass_power,
            distance_power,
            sigma_power,
            c_power,
        ),
        g_power=g_power,
        hbar_power=hbar_power,
        mass_power=mass_power,
        distance_power=distance_power,
        sigma_power=sigma_power,
        c_power=c_power,
    )


def generate_blind_correction_library(
    max_coupling_order: int = 2,
    max_sigma_power: int = 4,
) -> tuple[CorrectionTerm, ...]:
    terms = []
    for g_power in range(max_coupling_order + 1):
        for hbar_power in range(max_coupling_order + 1 - g_power):
            sigma_start = 1 if g_power == 0 and hbar_power == 0 else 0
            for sigma_power in range(sigma_start, max_sigma_power + 1):
                if g_power == 0 and hbar_power == 0 and sigma_power == 0:
                    continue
                mass_power = g_power - hbar_power
                c_power = -(2 * g_power + hbar_power)
                distance_power = -(g_power + hbar_power + sigma_power)
                expression = _format_dimensionless_term_expression(
                    g_power,
                    hbar_power,
                    mass_power,
                    distance_power,
                    sigma_power,
                    c_power,
                )
                terms.append(
                    make_correction_term(
                        expression,
                        g_power=g_power,
                        hbar_power=hbar_power,
                        mass_power=mass_power,
                        distance_power=distance_power,
                        sigma_power=sigma_power,
                        c_power=c_power,
                        expression=expression,
                    )
                )
    return tuple(terms)


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


CORRECTION_LIBRARY: tuple[CorrectionTerm, ...] = (
    make_correction_term(
        "pn_classical",
        g_power=1,
        hbar_power=0,
        mass_power=1,
        distance_power=-1,
        sigma_power=0,
        c_power=-2,
        expression="G*M/(r*c^2)",
    ),
    make_correction_term(
        "quantum_eft",
        g_power=1,
        hbar_power=1,
        mass_power=0,
        distance_power=-2,
        sigma_power=0,
        c_power=-3,
        expression="G*hbar/(r^2*c^3)",
    ),
    make_correction_term(
        "smearing_pn_cross",
        g_power=1,
        hbar_power=0,
        mass_power=1,
        distance_power=-3,
        sigma_power=2,
        c_power=-2,
        expression="G*sigma^2*M/(r^3*c^2)",
    ),
    make_correction_term(
        "sigma_over_r",
        g_power=0,
        hbar_power=0,
        mass_power=0,
        distance_power=-1,
        sigma_power=1,
        c_power=0,
        expression="sigma/r",
    ),
    make_correction_term(
        "second_order_pn",
        g_power=2,
        hbar_power=0,
        mass_power=2,
        distance_power=-2,
        sigma_power=0,
        c_power=-4,
        expression="G^2*M^2/(r^2*c^4)",
    ),
)


def predict_dataset(
    formula: CandidateFormula | CorrectionFormula,
    dataset: dict[str, object],
    physics_config: simulation.OracleConfig | None = None,
) -> PredictionBundle:
    physics_config = physics_config or getattr(formula, "physics_config", simulation.OracleConfig())
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
            hbar=physics_config.hbar,
            speed_of_light=physics_config.speed_of_light,
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
    prediction = predict_dataset(formula, dataset, config.oracle_config)
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


def correction_limit_penalty(formula: CorrectionFormula, config: SearchConfig) -> float:
    if not formula.terms:
        return 0.0
    samples = (
        [0.08, 0.07, 6.0, 0.2, 0.2, 1.0, 0.4, 1.5],
        [0.12, 0.09, 8.0, 0.3, 0.3, 1.0, 0.6, 2.0],
    )
    penalties = []
    for input_row in samples:
        total = 1.0
        distance = input_row[2]
        for coefficient, term in zip(formula.coefficients, formula.terms):
            total += coefficient * term.value(input_row, distance, config.oracle_config)
        penalties.append(max(0.0, -total) ** 2)
    return _mean(penalties)


def score_correction_formula(
    formula: CorrectionFormula,
    dataset: dict[str, object],
    config: SearchConfig,
) -> tuple[float, float, float, float, float]:
    prediction = predict_dataset(formula, dataset, config.oracle_config)
    gravity_error = _normalized_mse(prediction.gravity_targets, dataset["gravity_targets"])
    quantum_error = _normalized_mse(prediction.quantum_targets, dataset["quantum_targets"])
    asymptotic_penalty = correction_limit_penalty(formula, config)
    complexity = formula.complexity
    unified_score = (
        config.gravity_weight * gravity_error
        + config.quantum_weight * quantum_error
        + config.limit_weight * asymptotic_penalty
        + config.complexity_weight * complexity
    )
    return unified_score, gravity_error, quantum_error, asymptotic_penalty, complexity


def score_correction_formula_across_datasets(
    formula: CorrectionFormula,
    datasets: Sequence[dict[str, object]],
    config: SearchConfig,
) -> tuple[float, float, float, float, float]:
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    for dataset in datasets:
        metrics = score_correction_formula(formula, dataset, config)
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


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for pivot in range(size):
        max_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        augmented[pivot], augmented[max_row] = augmented[max_row], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        if abs(pivot_value) < 1e-18:
            raise ValueError("Singular matrix encountered while solving correction coefficients.")

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


def _correction_design_matrix(
    dataset: dict[str, object],
    terms: tuple[CorrectionTerm, ...],
    config: SearchConfig,
) -> list[list[float]]:
    matrix = []
    for input_row, distance_row in zip(dataset["inputs"], dataset["branch_distances"]):
        for distance in distance_row:
            matrix.append([term.value(input_row, distance, config.oracle_config) for term in terms])
    return matrix


def _flatten_correction_targets(dataset: dict[str, object]) -> list[float]:
    return [value - 1.0 for row in dataset["branch_total_correction_factors"] for value in row]


def fit_correction_formula(
    train_dataset: dict[str, object],
    terms: tuple[CorrectionTerm, ...],
    config: SearchConfig,
) -> CorrectionFormula:
    if not terms:
        return CorrectionFormula(terms=(), coefficients=(), physics_config=config.oracle_config, complexity=1.0)

    design = _correction_design_matrix(train_dataset, terms, config)
    target = _flatten_correction_targets(train_dataset)
    size = len(terms)
    column_scales = [max(_rms([row[column] for row in design]), 1e-18) for column in range(size)]
    target_scale = max(_rms(target), 1e-18)

    scaled_design = [[value / scale for value, scale in zip(row, column_scales)] for row in design]
    scaled_target = [value / target_scale for value in target]

    gram = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    for row, y_value in zip(scaled_design, scaled_target):
        for i in range(size):
            rhs[i] += row[i] * y_value
            for j in range(size):
                gram[i][j] += row[i] * row[j]

    for i in range(size):
        gram[i][i] += max(config.ridge, 1e-12)

    scaled_coefficients = _solve_linear_system(gram, rhs)
    coefficients = tuple(
        target_scale * coefficient / scale
        for coefficient, scale in zip(scaled_coefficients, column_scales)
    )
    formula = CorrectionFormula(
        terms=terms,
        coefficients=coefficients,
        physics_config=config.oracle_config,
        complexity=1.0 + 0.2 * len(terms),
    )
    return canonicalize_correction_formula(formula)


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
    train_dataset = simulation.make_dataset(
        config.num_train,
        seed=config.seed,
        regime=config.train_regime,
        config=config.oracle_config,
    )
    val_datasets = [
        simulation.make_dataset(
            config.num_val,
            seed=config.seed + 1 + 97 * regime_index,
            regime=regime,
            config=config.oracle_config,
        )
        for regime_index, regime in enumerate(config.validation_regimes)
    ]
    return run_uniqueness_sweep_on_datasets(train_dataset, val_datasets, config)


def search_best_correction_formula(
    train_dataset: dict[str, object],
    val_datasets: Sequence[dict[str, object]],
    config: SearchConfig,
    correction_library: Sequence[CorrectionTerm] | None = None,
    max_subset_size: int | None = None,
) -> tuple[CorrectionFormula, tuple[float, float, float, float, float]]:
    correction_library = tuple(correction_library or CORRECTION_LIBRARY)
    max_subset_size = min(max_subset_size or len(correction_library), len(correction_library))
    best_formula = CorrectionFormula(terms=(), coefficients=(), physics_config=config.oracle_config, complexity=1.0)
    best_metrics = score_correction_formula_across_datasets(best_formula, val_datasets, config)

    for subset_size in range(1, max_subset_size + 1):
        for subset in itertools.combinations(correction_library, subset_size):
            formula = fit_correction_formula(train_dataset, subset, config)
            metrics = score_correction_formula_across_datasets(formula, val_datasets, config)
            if metrics[0] < best_metrics[0] - 1e-15:
                best_formula = formula
                best_metrics = metrics
            elif abs(metrics[0] - best_metrics[0]) <= 1e-15 and formula.complexity < best_formula.complexity:
                best_formula = formula
                best_metrics = metrics

    return best_formula, best_metrics


def rank_correction_subsets(
    train_dataset: dict[str, object],
    val_datasets: Sequence[dict[str, object]],
    config: SearchConfig,
    correction_library: Sequence[CorrectionTerm] | None = None,
    *,
    max_subset_size: int | None = None,
    top_n: int | None = None,
    coefficient_key: str = "name",
) -> list[CorrectionRankingEntry]:
    correction_library = tuple(correction_library or CORRECTION_LIBRARY)
    max_subset_size = min(max_subset_size or len(correction_library), len(correction_library))
    best_by_signature: dict[str, CorrectionRankingEntry] = {}

    for subset_size in range(1, max_subset_size + 1):
        for subset in itertools.combinations(correction_library, subset_size):
            formula = fit_correction_formula(train_dataset, subset, config)
            if not formula.terms:
                continue
            unified_score, gravity_error, quantum_error, asymptotic_penalty, complexity = (
                score_correction_formula_across_datasets(formula, val_datasets, config)
            )
            entry = CorrectionRankingEntry(
                formula=formula,
                unified_score=unified_score,
                gravity_error=gravity_error,
                quantum_error=quantum_error,
                limit_penalty=asymptotic_penalty,
                complexity_penalty=complexity,
                selected_term_names=tuple(term.name for term in formula.terms),
                coefficients=formula.coefficient_map(correction_library, key=coefficient_key),
            )
            signature = entry.formula_text
            previous = best_by_signature.get(signature)
            if previous is None or (
                entry.unified_score,
                entry.complexity_penalty,
                entry.selected_term_names,
            ) < (
                previous.unified_score,
                previous.complexity_penalty,
                previous.selected_term_names,
            ):
                best_by_signature[signature] = entry

    rankings = list(best_by_signature.values())
    rankings.sort(
        key=lambda entry: (
            entry.unified_score,
            entry.complexity_penalty,
            entry.selected_term_names,
        )
    )
    if top_n is None:
        return rankings
    return rankings[:top_n]


def _default_eft_config() -> SearchConfig:
    return SearchConfig(
        train_regime="eft_sensitive",
        validation_regimes=("eft_sensitive_compact", "eft_sensitive_wide"),
        oracle_config=AMPLIFIED_EFT_CONFIG,
    )


def _run_correction_recovery(
    config: SearchConfig,
    correction_library: Sequence[CorrectionTerm],
    *,
    max_subset_size: int | None = None,
    coefficient_key: str = "name",
) -> CorrectionRecoveryResult:
    correction_library = tuple(correction_library)
    config = config or SearchConfig(
        train_regime="eft_sensitive",
        validation_regimes=("eft_sensitive_compact", "eft_sensitive_wide"),
        oracle_config=AMPLIFIED_EFT_CONFIG,
    )
    t0 = time.time()

    zero_formula = CorrectionFormula(terms=(), coefficients=(), physics_config=config.oracle_config, complexity=1.0)
    best_formula = zero_formula
    best_metrics = None
    best_val_datasets = None
    search_rounds = 0

    while True:
        round_seed = config.seed + 1000 * search_rounds
        train_dataset = simulation.make_dataset(
            config.num_train,
            seed=round_seed,
            regime=config.train_regime,
            config=config.oracle_config,
        )
        val_datasets = [
            simulation.make_dataset(
                config.num_val,
                seed=round_seed + 1 + 97 * regime_index,
                regime=regime,
                config=config.oracle_config,
            )
            for regime_index, regime in enumerate(config.validation_regimes)
        ]

        if search_rounds == 0:
            best_metrics = score_correction_formula_across_datasets(zero_formula, val_datasets, config)
            best_val_datasets = val_datasets

        formula, metrics = search_best_correction_formula(
            train_dataset,
            val_datasets,
            config,
            correction_library=correction_library,
            max_subset_size=max_subset_size,
        )
        if best_metrics is None or metrics[0] < best_metrics[0] - 1e-15:
            best_formula = formula
            best_metrics = metrics
            best_val_datasets = val_datasets
        elif abs(metrics[0] - best_metrics[0]) <= 1e-15 and formula.complexity < best_formula.complexity:
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
        if not best_formula.terms
        else score_correction_formula_across_datasets(zero_formula, best_val_datasets, config)[0]
    )
    reference_dataset = best_val_datasets[0]
    prediction = predict_dataset(best_formula, reference_dataset, config.oracle_config)
    validation_summary = "heldout validation: " + ", ".join(dataset["regime"] for dataset in best_val_datasets)
    plot_path = save_diagnostic_plot(
        reference_dataset,
        prediction,
        best_formula.formula_text(),
        validation_summary,
        config.output_dir,
    )
    unified_score, gravity_error, quantum_error, _, _ = best_metrics
    return CorrectionRecoveryResult(
        unified_score=unified_score,
        gravity_error=gravity_error,
        quantum_error=quantum_error,
        zero_formula_score=zero_formula_score,
        formula_text=best_formula.formula_text(),
        plot_path=plot_path,
        coefficients=best_formula.coefficient_map(correction_library, key=coefficient_key),
        selected_term_names=tuple(term.name for term in best_formula.terms),
        validation_summary=validation_summary,
        num_validation_regimes=len(best_val_datasets),
        search_rounds=search_rounds,
        duration_seconds=time.time() - t0,
    )


def run_correction_recovery_experiment(config: SearchConfig | None = None) -> CorrectionRecoveryResult:
    return _run_correction_recovery(config or _default_eft_config(), CORRECTION_LIBRARY)


def run_blind_correction_recovery_experiment(
    config: SearchConfig | None = None,
    *,
    max_subset_size: int = 3,
    max_coupling_order: int = 2,
    max_sigma_power: int = 4,
) -> CorrectionRecoveryResult:
    correction_library = generate_blind_correction_library(
        max_coupling_order=max_coupling_order,
        max_sigma_power=max_sigma_power,
    )
    return _run_correction_recovery(
        config or _default_eft_config(),
        correction_library,
        max_subset_size=max_subset_size,
        coefficient_key="expression",
    )


def run_blind_degeneracy_analysis(
    config: SearchConfig | None = None,
    *,
    max_subset_size: int = 3,
    max_coupling_order: int = 2,
    max_sigma_power: int = 4,
    top_n: int = 5,
) -> BlindDegeneracyReport:
    config = config or _default_eft_config()
    correction_library = generate_blind_correction_library(
        max_coupling_order=max_coupling_order,
        max_sigma_power=max_sigma_power,
    )
    t0 = time.time()
    train_dataset = simulation.make_dataset(
        config.num_train,
        seed=config.seed,
        regime=config.train_regime,
        config=config.oracle_config,
    )
    val_datasets = [
        simulation.make_dataset(
            config.num_val,
            seed=config.seed + 1 + 97 * regime_index,
            regime=regime,
            config=config.oracle_config,
        )
        for regime_index, regime in enumerate(config.validation_regimes)
    ]
    rankings = rank_correction_subsets(
        train_dataset,
        val_datasets,
        config,
        correction_library=correction_library,
        max_subset_size=max_subset_size,
        top_n=top_n,
        coefficient_key="expression",
    )
    top_formula = rankings[0].formula
    prediction = predict_dataset(top_formula, val_datasets[0], config.oracle_config)
    validation_summary = "heldout validation: " + ", ".join(dataset["regime"] for dataset in val_datasets)
    plot_path = save_diagnostic_plot(
        val_datasets[0],
        prediction,
        top_formula.formula_text(),
        validation_summary,
        config.output_dir,
    )
    zero_formula = CorrectionFormula(terms=(), coefficients=(), physics_config=config.oracle_config, complexity=1.0)
    zero_formula_score = score_correction_formula_across_datasets(zero_formula, val_datasets, config)[0]
    total_subsets = sum(
        math.comb(len(correction_library), subset_size)
        for subset_size in range(1, min(max_subset_size, len(correction_library)) + 1)
    )
    runner_up_margin = (
        rankings[1].unified_score - rankings[0].unified_score
        if len(rankings) >= 2
        else float("inf")
    )
    return BlindDegeneracyReport(
        top_rankings=tuple(rankings),
        runner_up_margin=runner_up_margin,
        total_subsets=total_subsets,
        zero_formula_score=zero_formula_score,
        plot_path=plot_path,
        validation_summary=validation_summary,
        duration_seconds=time.time() - t0,
    )


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
        train_dataset = simulation.make_dataset(
            config.num_train,
            seed=round_seed,
            regime=config.train_regime,
            config=config.oracle_config,
        )
        val_datasets = [
            simulation.make_dataset(
                config.num_val,
                seed=round_seed + 1 + 97 * regime_index,
                regime=regime,
                config=config.oracle_config,
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
    prediction = predict_dataset(best_formula, reference_dataset, config.oracle_config)
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
    parser = argparse.ArgumentParser(description="Run unified gravity/quantum recovery experiments.")
    parser.add_argument(
        "--mode",
        choices=("smearing", "eft", "blind", "degeneracy"),
        default="smearing",
        help="Choose the original smearing search, the curated EFT recovery, the blind dimensional search, or the blind degeneracy ranking.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for diagnostic artifacts such as diagnostics.svg.",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
        help="Per-run search budget in seconds.",
    )
    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=3,
        help="Maximum subset size for blind correction searches.",
    )
    parser.add_argument(
        "--max-coupling-order",
        type=int,
        default=2,
        help="Maximum total G/hbar coupling order for blind correction enumeration.",
    )
    parser.add_argument(
        "--max-sigma-power",
        type=int,
        default=4,
        help="Maximum sigma power for blind correction enumeration.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many ranked blind subsets to report in degeneracy mode.",
    )
    args = parser.parse_args()
    base_config = SearchConfig(output_dir=args.output_dir, time_budget_seconds=args.time_budget_seconds)

    if args.mode in {"eft", "blind", "degeneracy"}:
        eft_config = SearchConfig(
            output_dir=args.output_dir,
            time_budget_seconds=args.time_budget_seconds,
            train_regime="eft_sensitive",
            validation_regimes=("eft_sensitive_compact", "eft_sensitive_wide"),
            oracle_config=AMPLIFIED_EFT_CONFIG,
        )
        if args.mode == "degeneracy":
            report = run_blind_degeneracy_analysis(
                eft_config,
                max_subset_size=args.max_subset_size,
                max_coupling_order=args.max_coupling_order,
                max_sigma_power=args.max_sigma_power,
                top_n=args.top_n,
            )
            print("---")
            print(f"total_subsets:    {report.total_subsets}")
            print(f"runner_up_margin: {report.runner_up_margin:.6f}")
            print(f"zero_formula:     {report.zero_formula_score:.6f}")
            print(f"validation:       {report.validation_summary}")
            print(f"seconds:          {report.duration_seconds:.2f}")
            print(f"plot_path:        {report.plot_path}")
            for rank, entry in enumerate(report.top_rankings, start=1):
                print(
                    f"rank_{rank}:         score={entry.unified_score:.6f} "
                    f"terms={', '.join(entry.selected_term_names)}"
                )
                print(f"rank_{rank}_formula: {entry.formula_text}")
            return
        if args.mode == "blind":
            result = run_blind_correction_recovery_experiment(
                eft_config,
                max_subset_size=args.max_subset_size,
                max_coupling_order=args.max_coupling_order,
                max_sigma_power=args.max_sigma_power,
            )
        else:
            result = run_correction_recovery_experiment(eft_config)
        print("---")
        print(f"unified_score:    {result.unified_score:.6f}")
        print(f"gravity_error:    {result.gravity_error:.6f}")
        print(f"quantum_error:    {result.quantum_error:.6f}")
        print(f"zero_formula:     {result.zero_formula_score:.6f}")
        print(f"validation_sets:  {result.num_validation_regimes}")
        print(f"validation:       {result.validation_summary}")
        print(f"search_rounds:    {result.search_rounds}")
        print(f"time_budget_s:    {args.time_budget_seconds:.2f}")
        print(f"seconds:          {result.duration_seconds:.2f}")
        print(f"plot_path:        {result.plot_path}")
        if args.mode == "blind":
            print(f"max_subset_size:  {args.max_subset_size}")
            print(f"max_coupling:     {args.max_coupling_order}")
            print(f"max_sigma_power:  {args.max_sigma_power}")
        print(f"selected_terms:   {', '.join(result.selected_term_names) if result.selected_term_names else '(none)'}")
        print(f"formula:          {result.formula_text}")
        for name, value in result.coefficients.items():
            print(f"coef_{name}:      {value:.6f}")
        return

    result = run_experiment(base_config)
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
    print(f"time_budget_s:    {args.time_budget_seconds:.2f}")
    print(f"seconds:          {result.duration_seconds:.2f}")
    print(f"plot_path:        {result.plot_path}")
    print(f"formula:          {result.formula_text}")


if __name__ == "__main__":
    main()
