from __future__ import annotations

import argparse
import cmath
import json
import math
import random
from dataclasses import asdict, dataclass

BRANCHES = ("LL", "LR", "RL", "RR")
HADAMARD_2Q = (
    (0.5, 0.5, 0.5, 0.5),
    (0.5, -0.5, 0.5, -0.5),
    (0.5, 0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5, 0.5),
)
UNIFORM_PROBABILITIES = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class OracleConfig:
    gravitational_constant: float = 1.0
    hbar: float = 1.0
    speed_of_light: float = 8.0
    min_distance: float = 0.25


@dataclass(frozen=True)
class OracleSample:
    m1: float
    m2: float
    base_distance: float
    delta1: float
    delta2: float
    interaction_time: float
    wavepacket_width: float
    coherence_length: float

    def as_input_vector(self) -> list[float]:
        return [
            self.m1,
            self.m2,
            self.base_distance,
            self.delta1,
            self.delta2,
            self.interaction_time,
            self.wavepacket_width,
            self.coherence_length,
        ]


@dataclass(frozen=True)
class QuantumOutputs:
    branch_redshift_factors: tuple[float, float, float, float]
    branch_phases: tuple[float, float, float, float]
    recombined_probabilities: tuple[float, float, float, float]
    concurrence: float
    visibility: float


@dataclass(frozen=True)
class OracleOutputs:
    branch_distances: tuple[float, float, float, float]
    branch_effective_distances: tuple[float, float, float, float]
    branch_potentials: tuple[float, float, float, float]
    branch_forces: tuple[float, float, float, float]
    branch_redshift_factors: tuple[float, float, float, float]
    branch_phases: tuple[float, float, float, float]
    recombined_probabilities: tuple[float, float, float, float]
    mean_potential: float
    mean_force: float
    force_spread: float
    concurrence: float
    visibility: float


def _validate_sample(sample: OracleSample, config: OracleConfig) -> None:
    nearest_distance = sample.base_distance - 0.5 * (sample.delta1 + sample.delta2)
    if nearest_distance <= config.min_distance:
        raise ValueError(
            "Sample geometry is invalid: nearest branch distance must stay above min_distance."
        )
    if sample.wavepacket_width <= 0.0:
        raise ValueError("Wavepacket width must be positive.")
    if sample.coherence_length <= 0.0:
        raise ValueError("Coherence length must be positive.")


def _mean(values: tuple[float, ...] | list[float]) -> float:
    return sum(values) / len(values)


def _std(values: tuple[float, ...] | list[float]) -> float:
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def branch_distances(sample: OracleSample, config: OracleConfig | None = None) -> tuple[float, float, float, float]:
    config = config or OracleConfig()
    _validate_sample(sample, config)

    x1_left = -0.5 * sample.delta1
    x1_right = 0.5 * sample.delta1
    x2_left = sample.base_distance - 0.5 * sample.delta2
    x2_right = sample.base_distance + 0.5 * sample.delta2

    return (
        abs(x2_left - x1_left),
        abs(x2_right - x1_left),
        abs(x2_left - x1_right),
        abs(x2_right - x1_right),
    )


def effective_distance(distance: float, wavepacket_width: float) -> float:
    return math.sqrt(distance * distance + wavepacket_width * wavepacket_width)


def quantum_observables_from_branch_dynamics(
    branch_potentials: tuple[float, float, float, float] | list[float],
    branch_forces: tuple[float, float, float, float] | list[float],
    interaction_time: float,
    total_mass: float,
    wavepacket_width: float,
    coherence_length: float,
    hbar: float = 1.0,
    speed_of_light: float = 8.0,
) -> QuantumOutputs:
    branch_redshift_factors = []
    phases = []
    compactness_scale = max(total_mass * speed_of_light * speed_of_light, 1e-9)

    for potential in branch_potentials:
        reduced_potential = potential / compactness_scale
        redshift = 1.0 + reduced_potential - 0.5 * reduced_potential * reduced_potential
        branch_redshift_factors.append(redshift)
        phase = (-potential + 0.5 * potential * potential / compactness_scale) * interaction_time / hbar
        phases.append(phase)

    amplitudes = tuple(cmath.exp(1j * phase) / 2.0 for phase in phases)
    coherent = []
    for row in HADAMARD_2Q:
        amplitude = sum(weight * branch for weight, branch in zip(row, amplitudes))
        coherent.append(float((amplitude.conjugate() * amplitude).real))

    pure_concurrence = 2.0 * abs(amplitudes[0] * amplitudes[3] - amplitudes[1] * amplitudes[2])
    force_spread = _std(branch_forces)
    dephasing_argument = force_spread * interaction_time * wavepacket_width / (hbar * coherence_length)
    visibility = math.exp(-(dephasing_argument ** 2))
    probabilities = tuple(
        visibility * coherent_probability + (1.0 - visibility) * uniform_probability
        for coherent_probability, uniform_probability in zip(coherent, UNIFORM_PROBABILITIES)
    )
    total_probability = sum(probabilities)
    normalized_probabilities = tuple(probability / total_probability for probability in probabilities)

    return QuantumOutputs(
        branch_redshift_factors=tuple(branch_redshift_factors),
        branch_phases=tuple(phases),
        recombined_probabilities=normalized_probabilities,
        concurrence=float(max(0.0, min(1.0, visibility * pure_concurrence))),
        visibility=float(max(0.0, min(1.0, visibility))),
    )


def oracle(sample: OracleSample, config: OracleConfig | None = None) -> OracleOutputs:
    config = config or OracleConfig()
    distances = branch_distances(sample, config)
    effective_distances = tuple(
        effective_distance(distance, sample.wavepacket_width)
        for distance in distances
    )

    mu = sample.m1 * sample.m2
    total_mass = sample.m1 + sample.m2
    g = config.gravitational_constant
    c_sq = config.speed_of_light * config.speed_of_light
    compactness_scale = g * total_mass / (2.0 * c_sq)

    potentials = []
    forces = []
    for distance, rho in zip(distances, effective_distances):
        inverse_rho = 1.0 / rho
        inverse_rho_sq = inverse_rho * inverse_rho
        potential = -g * mu * (inverse_rho + compactness_scale * inverse_rho_sq)
        force = g * mu * distance * (
            (1.0 / (rho ** 3)) + (2.0 * compactness_scale / (rho ** 4))
        )
        potentials.append(potential)
        forces.append(force)

    quantum = quantum_observables_from_branch_dynamics(
        potentials,
        forces,
        interaction_time=sample.interaction_time,
        total_mass=total_mass,
        wavepacket_width=sample.wavepacket_width,
        coherence_length=sample.coherence_length,
        hbar=config.hbar,
        speed_of_light=config.speed_of_light,
    )

    return OracleOutputs(
        branch_distances=distances,
        branch_effective_distances=effective_distances,
        branch_potentials=tuple(potentials),
        branch_forces=tuple(forces),
        branch_redshift_factors=quantum.branch_redshift_factors,
        branch_phases=quantum.branch_phases,
        recombined_probabilities=quantum.recombined_probabilities,
        mean_potential=_mean(potentials),
        mean_force=_mean(forces),
        force_spread=_std(forces),
        concurrence=quantum.concurrence,
        visibility=quantum.visibility,
    )


def make_dataset(
    num_samples: int,
    seed: int = 0,
    config: OracleConfig | None = None,
) -> dict[str, object]:
    config = config or OracleConfig()
    rng = random.Random(seed)

    samples: list[OracleSample] = []
    while len(samples) < num_samples:
        sample = OracleSample(
            m1=rng.uniform(0.5, 2.0),
            m2=rng.uniform(0.5, 2.0),
            base_distance=rng.uniform(1.5, 4.5),
            delta1=rng.uniform(0.2, 1.0),
            delta2=rng.uniform(0.2, 1.0),
            interaction_time=rng.uniform(0.3, 1.8),
            wavepacket_width=rng.uniform(0.1, 0.6),
            coherence_length=rng.uniform(0.6, 2.5),
        )
        try:
            _validate_sample(sample, config)
        except ValueError:
            continue
        samples.append(sample)

    outputs = [oracle(sample, config) for sample in samples]
    return {
        "samples": samples,
        "inputs": [sample.as_input_vector() for sample in samples],
        "branch_distances": [list(out.branch_distances) for out in outputs],
        "branch_effective_distances": [list(out.branch_effective_distances) for out in outputs],
        "branch_potentials": [list(out.branch_potentials) for out in outputs],
        "branch_forces": [list(out.branch_forces) for out in outputs],
        "gravity_targets": [
            [out.mean_potential, out.mean_force, out.force_spread]
            for out in outputs
        ],
        "quantum_targets": [
            list(out.recombined_probabilities) + [out.concurrence, out.visibility]
            for out in outputs
        ],
    }


def _preview_payload(num_samples: int, seed: int) -> dict[str, object]:
    dataset = make_dataset(num_samples=num_samples, seed=seed)
    first = oracle(dataset["samples"][0])
    return {
        "first_sample": asdict(dataset["samples"][0]),
        "first_oracle": {
            "branch_distances": list(first.branch_distances),
            "branch_effective_distances": list(first.branch_effective_distances),
            "branch_potentials": list(first.branch_potentials),
            "branch_forces": list(first.branch_forces),
            "branch_redshift_factors": list(first.branch_redshift_factors),
            "branch_phases": list(first.branch_phases),
            "recombined_probabilities": list(first.recombined_probabilities),
            "mean_potential": first.mean_potential,
            "mean_force": first.mean_force,
            "force_spread": first.force_spread,
            "concurrence": first.concurrence,
            "visibility": first.visibility,
        },
        "dataset_shapes": {
            "inputs": [len(dataset["inputs"]), len(dataset["inputs"][0])],
            "branch_distances": [len(dataset["branch_distances"]), len(dataset["branch_distances"][0])],
            "branch_effective_distances": [len(dataset["branch_effective_distances"]), len(dataset["branch_effective_distances"][0])],
            "branch_potentials": [len(dataset["branch_potentials"]), len(dataset["branch_potentials"][0])],
            "branch_forces": [len(dataset["branch_forces"]), len(dataset["branch_forces"][0])],
            "gravity_targets": [len(dataset["gravity_targets"]), len(dataset["gravity_targets"][0])],
            "quantum_targets": [len(dataset["quantum_targets"]), len(dataset["quantum_targets"][0])],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the fixed quantum-plus-gravity oracle.")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(_preview_payload(num_samples=args.samples, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
