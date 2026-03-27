# Future Work: Theory Discrimination via Automated Symbolic Regression

## Motivation

Phases 1-4 are complete. The blind search robustly recovers the Donoghue EFT correction structure from a 29-term dimensional-analysis search space. The methodology is proven. The oracle is solved.

The next genuinely interesting result requires changing what the oracle represents.

## Proposal: Multi-Oracle Theory Discrimination

Build three oracles encoding competing theories of quantum gravity. Run the same blind search on each. Compare what the search recovers.

### Oracle A: Donoghue EFT (current)

Branch-dependent potential with perturbative corrections:

```
V_ab = -G m1 m2 erf(r_ab/(2*sigma)) / r_ab * [1 + 3GM/(r_ab*c^2) + (41/(10*pi))*G*hbar/(r_ab^2*c^3)]
```

Each branch sees a different potential depending on the spatial configuration. This generates gravitationally-induced entanglement through branch-dependent phase accumulation.

### Oracle B: Semiclassical Gravity (Schrodinger-Newton)

The gravitational field is sourced by the expectation value of the mass distribution, not the branch-specific position. Every branch sees the same mean-field potential:

```
V_ab = -G m1 m2 erf(r_mean/(2*sigma)) / r_mean    (same for all branches)
```

This kills gravitationally-induced entanglement entirely. Concurrence stays zero. The quantum signal comes only from the initial superposition, not from gravity.

### Oracle C: Diosi-Penrose Gravitational Decoherence

The potential is branch-dependent (like Oracle A), but gravity causes additional decoherence proportional to the gravitational self-energy difference between branches:

```
Gamma_decoherence ~ G * (m1^2 * delta1^2 + m2^2 * delta2^2) / hbar
```

This is a new decoherence channel that reduces visibility faster than coherence_length alone. The correction structure includes both Donoghue terms and a decoherence rate with no analogue in Oracle A.

## The Experiment

### Step 1: Implement Oracles B and C

Oracle B requires replacing per-branch potentials with the mean-field potential in `oracle()`. Small change.

Oracle C requires adding a decoherence rate term to the visibility calculation in `quantum_observables_from_branch_dynamics()`. Small change.

Both use the existing `simulation.py` infrastructure. The amplified config works for all three.

### Step 2: Blind Search on Each Oracle

Run `--mode blind` identically on each oracle. The search side (`train.py`) does not change. Each oracle produces a different recovered correction structure.

### Step 3: Cross-Oracle Scoring

Score each recovered formula against data from the other oracles. The result is a confusion matrix:

```
                    scored on Oracle A    scored on Oracle B    scored on Oracle C
found by Oracle A   0.000000              ???                   ???
found by Oracle B   ???                   0.000000              ???
found by Oracle C   ???                   ???                   0.000000
```

Large off-diagonal entries mean the theories are distinguishable. Small off-diagonal entries mean they are degenerate in this measurement setup.

### Step 4: Regime Sensitivity Analysis

Vary the oracle regimes (masses, distances, interaction times) and measure which parameter ranges maximize the off-diagonal divergence. The regime that produces the largest cross-oracle prediction error is the optimal experimental configuration for distinguishing the theories.

## What Makes This Interesting

1. It is not oracle recovery. The search answers an open question about which theories are experimentally distinguishable.
2. The cross-oracle confusion matrix is a novel artifact. Nobody has used automated symbolic regression for quantum gravity theory selection.
3. The regime sensitivity analysis produces experimental design guidance: which measurement configurations best distinguish semiclassical gravity from gravitational decoherence from EFT corrections.
4. The implementation is tractable. Each oracle is a small modification of the existing code. The search infrastructure is unchanged.

## What a Positive Result Looks Like

- The search recovers fundamentally different correction structures from different oracles.
- The confusion matrix shows large off-diagonal entries for at least one pair of theories.
- The regime analysis identifies specific parameter ranges (masses, distances, times) that maximize discrimination.
- The result is expressible as: "at masses m ~ X, separations r ~ Y, and interaction times t ~ Z, Oracle A predicts probability pattern P while Oracle C predicts Q, with divergence D."

## What a Negative Result Looks Like

- All three oracles produce indistinguishable correction structures in the blind search.
- The confusion matrix is nearly diagonal everywhere.
- This would mean the current measurement setup (two masses in superposition, Hadamard readout) lacks the resolving power to distinguish the theories, and a fundamentally different experimental design is needed.

A negative result is still informative: it tells you what this class of experiments cannot do.

## References

- Donoghue, J. F., "General relativity as an effective field theory," Phys. Rev. D 50, 3874 (1994)
- Diosi, L., "A universal master equation for the gravitational violation of quantum mechanics," Phys. Lett. A 120, 377 (1987)
- Penrose, R., "On gravity's role in quantum state reduction," Gen. Rel. Grav. 28, 581 (1996)
- Bahrami, M. et al., "The Schrodinger-Newton equation and its foundations," New J. Phys. 16, 115007 (2014)
- Bose, S. et al., "Spin Entanglement Witness for Quantum Gravity," Phys. Rev. Lett. 119, 240401 (2017)
- Aziz, J. and Howl, R., "Classical theories of gravity produce entanglement," Nature 646, 49 (2025)
