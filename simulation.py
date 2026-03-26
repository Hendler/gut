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


@dataclass(frozen=True)
class OracleConfig:
    gravitational_constant: float = 1.0
    hbar: float = 1.0
    min_distance: float = 0.25


@dataclass(frozen=True)
class OracleSample:
    m1: float
    m2: float
    base_distance: float
    delta1: float
    delta2: float
    interaction_time: float

    def as_input_vector(self) -> list[float]:
        return [
            self.m1,
            self.m2,
            self.base_distance,
            self.delta1,
            self.delta2,
            self.interaction_time,
        ]


@dataclass(frozen=True)
class QuantumOutputs:
    branch_phases: tuple[float, float, float, float]
    recombined_probabilities: tuple[float, float, float, float]
    concurrence: float


@dataclass(frozen=True)
class OracleOutputs:
    branch_distances: tuple[float, float, float, float]
    branch_potentials: tuple[float, float, float, float]
    branch_forces: tuple[float, float, float, float]
    branch_phases: tuple[float, float, float, float]
    recombined_probabilities: tuple[float, float, float, float]
    mean_potential: float
    mean_force: float
    concurrence: float


def _validate_sample(sample: OracleSample, config: OracleConfig) -> None:
    nearest_distance = sample.base_distance - 0.5 * (sample.delta1 + sample.delta2)
    if nearest_distance <= config.min_distance:
        raise ValueError(
            "Sample geometry is invalid: nearest branch distance must stay above min_distance."
        )


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


def quantum_observables_from_branch_potentials(
    branch_potentials: tuple[float, float, float, float] | list[float],
    interaction_time: float,
    hbar: float = 1.0,
) -> QuantumOutputs:
    phases = tuple(-potential * interaction_time / hbar for potential in branch_potentials)
    amplitudes = tuple(cmath.exp(1j * phase) / 2.0 for phase in phases)

    recombined = []
    for row in HADAMARD_2Q:
        amplitude = sum(weight * branch for weight, branch in zip(row, amplitudes))
        recombined.append(amplitude)

    probabilities = tuple(float((amp.conjugate() * amp).real) for amp in recombined)
    concurrence = 2.0 * abs(amplitudes[0] * amplitudes[3] - amplitudes[1] * amplitudes[2])
    return QuantumOutputs(
        branch_phases=phases,
        recombined_probabilities=probabilities,
        concurrence=float(max(0.0, min(1.0, concurrence))),
    )


def oracle(sample: OracleSample, config: OracleConfig | None = None) -> OracleOutputs:
    config = config or OracleConfig()
    distances = branch_distances(sample, config)
    mu = sample.m1 * sample.m2

    potentials = tuple(-config.gravitational_constant * mu / distance for distance in distances)
    forces = tuple(config.gravitational_constant * mu / (distance * distance) for distance in distances)
    quantum = quantum_observables_from_branch_potentials(
        potentials,
        interaction_time=sample.interaction_time,
        hbar=config.hbar,
    )
    return OracleOutputs(
        branch_distances=distances,
        branch_potentials=potentials,
        branch_forces=forces,
        branch_phases=quantum.branch_phases,
        recombined_probabilities=quantum.recombined_probabilities,
        mean_potential=sum(potentials) / len(potentials),
        mean_force=sum(forces) / len(forces),
        concurrence=quantum.concurrence,
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
        "branch_potentials": [list(out.branch_potentials) for out in outputs],
        "branch_forces": [list(out.branch_forces) for out in outputs],
        "gravity_targets": [[out.mean_potential, out.mean_force] for out in outputs],
        "quantum_targets": [
            list(out.recombined_probabilities) + [out.concurrence] for out in outputs
        ],
    }


def _preview_payload(num_samples: int, seed: int) -> dict[str, object]:
    dataset = make_dataset(num_samples=num_samples, seed=seed)
    first = oracle(dataset["samples"][0])
    return {
        "first_sample": asdict(dataset["samples"][0]),
        "first_oracle": {
            "branch_distances": list(first.branch_distances),
            "branch_potentials": list(first.branch_potentials),
            "branch_forces": list(first.branch_forces),
            "branch_phases": list(first.branch_phases),
            "recombined_probabilities": list(first.recombined_probabilities),
            "mean_potential": first.mean_potential,
            "mean_force": first.mean_force,
            "concurrence": first.concurrence,
        },
        "dataset_shapes": {
            "inputs": [len(dataset["inputs"]), len(dataset["inputs"][0])],
            "branch_distances": [len(dataset["branch_distances"]), len(dataset["branch_distances"][0])],
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
