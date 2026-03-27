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
    gravitational_constant: float = 6.67430e-11
    hbar: float = 1.054571817e-34
    speed_of_light: float = 299792458.0
    min_distance: float = 5.0e-6


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
    branch_smearing_factors: tuple[float, float, float, float]
    branch_post_newtonian_corrections: tuple[float, float, float, float]
    branch_quantum_eft_corrections: tuple[float, float, float, float]
    branch_total_correction_factors: tuple[float, float, float, float]
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
    if sample.m1 <= 0.0 or sample.m2 <= 0.0:
        raise ValueError("Masses must be positive.")
    if sample.interaction_time <= 0.0:
        raise ValueError("Interaction time must be positive.")
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


def _gaussian_smearing_factor(distance: float, wavepacket_width: float) -> float:
    sigma = max(wavepacket_width, 1e-18)
    return math.erf(distance / (2.0 * sigma))


def effective_distance(distance: float, wavepacket_width: float) -> float:
    smearing = max(_gaussian_smearing_factor(distance, wavepacket_width), 1e-18)
    return distance / smearing


def branch_correction_terms(
    sample: OracleSample,
    distance: float,
    config: OracleConfig | None = None,
) -> tuple[float, float, float]:
    config = config or OracleConfig()
    smearing = _gaussian_smearing_factor(distance, sample.wavepacket_width)
    post_newtonian = (
        3.0
        * config.gravitational_constant
        * (sample.m1 + sample.m2)
        / (distance * config.speed_of_light * config.speed_of_light)
    )
    quantum_eft = (
        (41.0 / (10.0 * math.pi))
        * config.gravitational_constant
        * config.hbar
        / (distance * distance * (config.speed_of_light ** 3))
    )
    return smearing, post_newtonian, quantum_eft


def branch_total_correction_factor(
    sample: OracleSample,
    distance: float,
    config: OracleConfig | None = None,
) -> float:
    _, post_newtonian, quantum_eft = branch_correction_terms(sample, distance, config)
    return 1.0 + post_newtonian + quantum_eft


def branch_potential(
    sample: OracleSample,
    distance: float,
    config: OracleConfig | None = None,
) -> float:
    config = config or OracleConfig()
    smearing, post_newtonian, quantum_eft = branch_correction_terms(sample, distance, config)
    correction = 1.0 + post_newtonian + quantum_eft
    return -config.gravitational_constant * sample.m1 * sample.m2 * smearing * correction / distance


def branch_force(
    sample: OracleSample,
    distance: float,
    config: OracleConfig | None = None,
) -> float:
    config = config or OracleConfig()
    step = min(1e-6, 0.02 * max(distance, config.min_distance))
    left_distance = max(config.min_distance + 1e-9, distance - step)
    right_distance = distance + step
    left_potential = branch_potential(sample, left_distance, config)
    right_potential = branch_potential(sample, right_distance, config)
    return abs(-(right_potential - left_potential) / (right_distance - left_distance))


def quantum_observables_from_branch_dynamics(
    branch_potentials: tuple[float, float, float, float] | list[float],
    branch_forces: tuple[float, float, float, float] | list[float],
    interaction_time: float,
    total_mass: float,
    wavepacket_width: float,
    coherence_length: float,
    path_separations: tuple[float, float] = (0.0, 0.0),
    hbar: float = 1.054571817e-34,
    speed_of_light: float = 299792458.0,
) -> QuantumOutputs:
    del branch_forces
    total_rest_energy = max(total_mass * speed_of_light * speed_of_light, 1e-30)
    branch_redshift_factors = []
    phases = []

    for potential in branch_potentials:
        branch_redshift_factors.append(max(0.0, 1.0 + potential / total_rest_energy))
        phases.append(-potential * interaction_time / hbar)

    amplitudes = tuple(cmath.exp(1j * phase) / 2.0 for phase in phases)
    coherent_probabilities = []
    for row in HADAMARD_2Q:
        amplitude = sum(weight * branch for weight, branch in zip(row, amplitudes))
        coherent_probabilities.append(float((amplitude.conjugate() * amplitude).real))

    pure_concurrence = 2.0 * abs(amplitudes[0] * amplitudes[3] - amplitudes[1] * amplitudes[2])
    rho_offdiag = amplitudes[0] * amplitudes[2].conjugate() + amplitudes[1] * amplitudes[3].conjugate()
    intrinsic_visibility = min(1.0, 2.0 * abs(rho_offdiag))
    overlap_exponent = -(
        path_separations[0] * path_separations[0] + path_separations[1] * path_separations[1]
    ) / (8.0 * coherence_length * coherence_length)
    overlap_visibility = math.exp(overlap_exponent)
    visibility = max(0.0, min(1.0, intrinsic_visibility * overlap_visibility))

    probabilities = tuple(
        overlap_visibility * coherent_probability + (1.0 - overlap_visibility) * uniform_probability
        for coherent_probability, uniform_probability in zip(coherent_probabilities, UNIFORM_PROBABILITIES)
    )
    total_probability = sum(probabilities)
    normalized_probabilities = tuple(probability / total_probability for probability in probabilities)

    return QuantumOutputs(
        branch_redshift_factors=tuple(branch_redshift_factors),
        branch_phases=tuple(phases),
        recombined_probabilities=normalized_probabilities,
        concurrence=float(max(0.0, min(1.0, pure_concurrence * overlap_visibility))),
        visibility=float(visibility),
    )


def oracle(sample: OracleSample, config: OracleConfig | None = None) -> OracleOutputs:
    config = config or OracleConfig()
    distances = branch_distances(sample, config)
    effective_distances = tuple(
        effective_distance(distance, sample.wavepacket_width)
        for distance in distances
    )
    correction_terms = [branch_correction_terms(sample, distance, config) for distance in distances]
    smearing_factors = tuple(term[0] for term in correction_terms)
    post_newtonian_corrections = tuple(term[1] for term in correction_terms)
    quantum_eft_corrections = tuple(term[2] for term in correction_terms)
    total_correction_factors = tuple(
        1.0 + post_newtonian + quantum_eft
        for _, post_newtonian, quantum_eft in correction_terms
    )
    potentials = [branch_potential(sample, distance, config) for distance in distances]
    forces = [branch_force(sample, distance, config) for distance in distances]

    quantum = quantum_observables_from_branch_dynamics(
        potentials,
        forces,
        interaction_time=sample.interaction_time,
        total_mass=sample.m1 + sample.m2,
        wavepacket_width=sample.wavepacket_width,
        coherence_length=sample.coherence_length,
        path_separations=(sample.delta1, sample.delta2),
        hbar=config.hbar,
        speed_of_light=config.speed_of_light,
    )

    return OracleOutputs(
        branch_distances=distances,
        branch_effective_distances=effective_distances,
        branch_smearing_factors=smearing_factors,
        branch_post_newtonian_corrections=post_newtonian_corrections,
        branch_quantum_eft_corrections=quantum_eft_corrections,
        branch_total_correction_factors=total_correction_factors,
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
    regime: str = "train",
) -> dict[str, object]:
    config = config or OracleConfig()
    rng = random.Random(seed)

    samples: list[OracleSample] = []
    while len(samples) < num_samples:
        sample = _sample_from_regime(rng, regime)
        try:
            _validate_sample(sample, config)
        except ValueError:
            continue
        samples.append(sample)

    outputs = [oracle(sample, config) for sample in samples]
    return {
        "regime": regime,
        "samples": samples,
        "inputs": [sample.as_input_vector() for sample in samples],
        "branch_distances": [list(out.branch_distances) for out in outputs],
        "branch_effective_distances": [list(out.branch_effective_distances) for out in outputs],
        "branch_smearing_factors": [list(out.branch_smearing_factors) for out in outputs],
        "branch_post_newtonian_corrections": [list(out.branch_post_newtonian_corrections) for out in outputs],
        "branch_quantum_eft_corrections": [list(out.branch_quantum_eft_corrections) for out in outputs],
        "branch_total_correction_factors": [list(out.branch_total_correction_factors) for out in outputs],
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


def _sample_from_regime(rng: random.Random, regime: str) -> OracleSample:
    if regime == "train":
        return OracleSample(
            m1=rng.uniform(0.8e-14, 3.0e-14),
            m2=rng.uniform(0.8e-14, 3.0e-14),
            base_distance=rng.uniform(1.8e-4, 6.0e-4),
            delta1=rng.uniform(2.0e-5, 1.2e-4),
            delta2=rng.uniform(2.0e-5, 1.2e-4),
            interaction_time=rng.uniform(0.2, 3.0),
            wavepacket_width=rng.uniform(5.0e-6, 4.0e-5),
            coherence_length=rng.uniform(6.0e-5, 2.5e-4),
        )
    if regime == "heldout_compact":
        return OracleSample(
            m1=rng.uniform(2.0e-14, 6.0e-14),
            m2=rng.uniform(2.0e-14, 6.0e-14),
            base_distance=rng.uniform(7.0e-5, 1.8e-4),
            delta1=rng.uniform(3.0e-5, 1.0e-4),
            delta2=rng.uniform(3.0e-5, 1.0e-4),
            interaction_time=rng.uniform(0.5, 5.0),
            wavepacket_width=rng.uniform(1.0e-5, 7.5e-5),
            coherence_length=rng.uniform(5.0e-5, 2.0e-4),
        )
    if regime == "heldout_decoherent":
        return OracleSample(
            m1=rng.uniform(0.8e-14, 3.5e-14),
            m2=rng.uniform(0.8e-14, 3.5e-14),
            base_distance=rng.uniform(1.8e-4, 5.0e-4),
            delta1=rng.uniform(4.0e-5, 1.5e-4),
            delta2=rng.uniform(4.0e-5, 1.5e-4),
            interaction_time=rng.uniform(0.5, 4.0),
            wavepacket_width=rng.uniform(8.0e-6, 5.0e-5),
            coherence_length=rng.uniform(1.5e-5, 6.0e-5),
        )
    if regime == "heldout_wide":
        return OracleSample(
            m1=rng.uniform(0.6e-14, 2.0e-14),
            m2=rng.uniform(0.6e-14, 2.0e-14),
            base_distance=rng.uniform(7.0e-4, 2.0e-3),
            delta1=rng.uniform(1.0e-5, 7.0e-5),
            delta2=rng.uniform(1.0e-5, 7.0e-5),
            interaction_time=rng.uniform(0.1, 2.5),
            wavepacket_width=rng.uniform(3.0e-6, 2.0e-5),
            coherence_length=rng.uniform(8.0e-5, 4.0e-4),
        )
    raise ValueError(f"Unknown regime: {regime}")


def _preview_payload(num_samples: int, seed: int, regime: str) -> dict[str, object]:
    dataset = make_dataset(num_samples=num_samples, seed=seed, regime=regime)
    first = oracle(dataset["samples"][0])
    return {
        "first_sample": asdict(dataset["samples"][0]),
        "first_oracle": {
            "branch_distances": list(first.branch_distances),
            "branch_effective_distances": list(first.branch_effective_distances),
            "branch_smearing_factors": list(first.branch_smearing_factors),
            "branch_post_newtonian_corrections": list(first.branch_post_newtonian_corrections),
            "branch_quantum_eft_corrections": list(first.branch_quantum_eft_corrections),
            "branch_total_correction_factors": list(first.branch_total_correction_factors),
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
            "regime": dataset["regime"],
            "inputs": [len(dataset["inputs"]), len(dataset["inputs"][0])],
            "branch_distances": [len(dataset["branch_distances"]), len(dataset["branch_distances"][0])],
            "branch_effective_distances": [len(dataset["branch_effective_distances"]), len(dataset["branch_effective_distances"][0])],
            "branch_smearing_factors": [len(dataset["branch_smearing_factors"]), len(dataset["branch_smearing_factors"][0])],
            "branch_post_newtonian_corrections": [len(dataset["branch_post_newtonian_corrections"]), len(dataset["branch_post_newtonian_corrections"][0])],
            "branch_quantum_eft_corrections": [len(dataset["branch_quantum_eft_corrections"]), len(dataset["branch_quantum_eft_corrections"][0])],
            "branch_total_correction_factors": [len(dataset["branch_total_correction_factors"]), len(dataset["branch_total_correction_factors"][0])],
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
    parser.add_argument("--regime", type=str, default="train")
    args = parser.parse_args()
    print(json.dumps(_preview_payload(num_samples=args.samples, seed=args.seed, regime=args.regime), indent=2))


if __name__ == "__main__":
    main()
