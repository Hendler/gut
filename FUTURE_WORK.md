# Future Work

## What We Recovered and What It Means

The blind search recovers:

```
V(r) = -G m1 m2 / r * erf(r/(2*sigma)) * [1 + 3G(m1+m2)/(r c^2) + (41/(10*pi))*G*hbar/(r^2 c^3)]
```

This is the leading-order result of Donoghue's EFT treatment of quantum general relativity. The key context:

- The Donoghue framework is not a single equation but the systematic treatment of GR as a low-energy effective field theory, organized by powers of `G E^2`.
- The `3G(m1+m2)/(rc^2)` term is the classical post-Newtonian (Schwarzschild) correction.
- The `(41/10pi) G hbar/(r^2 c^3)` term is the leading genuine quantum gravity correction — calculable within the EFT without knowing the UV completion.
- Higher-order operators (`R^2`, `R_{mu nu} R^{mu nu}`, etc.) are suppressed by further powers of `M_Pl` and are not included in our oracle.

What we demonstrated: automated symbolic regression can blindly recover both the classical and quantum corrections from simulated data, with exact coefficients, from a 29-term search space. The recovery is robust across seeds, dataset sizes, and search space enlargement.

What we did not demonstrate: anything beyond what Donoghue computed in 1994. The oracle encodes known physics and the search recovers it.

## Two Directions Forward

### Direction A: Climb the EFT Ladder

The Donoghue framework has a natural hierarchy of corrections. Our oracle currently includes only the leading terms. The next orders include:

- **Two-loop corrections**: additional `G^2` terms with known (partially computed) coefficients
- **Higher curvature operators**: `R^2` and `R_{mu nu} R^{mu nu}` contributions to the potential
- **Spin-dependent corrections**: terms that depend on the angular momentum of the masses
- **Non-analytic (long-range) corrections**: logarithmic terms `log(r)` that arise from infrared physics and are unambiguously calculable

The experiment: enrich the oracle with the next order of Donoghue corrections, expand the blind basis to include the corresponding dimensional monomials (including `log(r/r_0)` terms), and test whether the search can recover the two-loop structure.

This stays within known physics but tests the methodology at a harder level. The two-loop coefficients are partially computed in the literature, so there is a target to check against. The interesting question: at what order does the blind search fail to distinguish the correct terms from spurious alternatives? That failure point tells you the resolving power of this class of oracle.

### Direction B: Theory Discrimination

The Donoghue EFT is not the only candidate framework. Other approaches to quantum gravity make different predictions for the same experimental setup. The EFT framework itself is agnostic about the UV completion — the higher-order operators are left undetermined. Different UV completions (string theory, loop quantum gravity, etc.) would fill in those operators differently.

But even at leading order, there are competing *interpretations* of how gravity and quantum mechanics interact that make structurally different predictions. Build three oracles encoding these alternatives. Run the same blind search on each. Compare what the search recovers.

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

## Relationship Between the Two Directions

Direction A stays within the Donoghue EFT and tests whether the methodology scales to harder recovery problems. It produces a methodological result: "automated search can (or cannot) recover N-loop quantum gravity corrections from simulated data."

Direction B steps outside the EFT framework and tests whether different theoretical paradigms produce distinguishable experimental signatures. It produces a physics result: "these theories are (or are not) distinguishable in BMV-class experiments, and the optimal discriminating regime is X."

Direction A is safer (known target coefficients to check against). Direction B is more interesting (answers an open question). Both are tractable with the existing infrastructure.

The strongest combined result would be: recover the Donoghue two-loop corrections (Direction A), then show that the recovered EFT structure is distinguishable from semiclassical and decoherence alternatives (Direction B). That would demonstrate both the depth and breadth of the methodology.

## References

- Donoghue, J. F., "General relativity as an effective field theory: The leading quantum corrections," Phys. Rev. D 50, 3874 (1994)
- Donoghue, J. F., "Introduction to the effective field theory description of gravity," in Advanced School on Effective Theories, Almunecar, Spain (1995). arXiv:gr-qc/9512024
- Bjerrum-Bohr, N. E. J., Donoghue, J. F., and Holstein, B. R., "Quantum gravitational scattering at the Planckian energy scale," Phys. Rev. D 67, 084033 (2003)
- Diosi, L., "A universal master equation for the gravitational violation of quantum mechanics," Phys. Lett. A 120, 377 (1987)
- Penrose, R., "On gravity's role in quantum state reduction," Gen. Rel. Grav. 28, 581 (1996)
- Bahrami, M. et al., "The Schrodinger-Newton equation and its foundations," New J. Phys. 16, 115007 (2014)
- Bose, S. et al., "Spin Entanglement Witness for Quantum Gravity," Phys. Rev. Lett. 119, 240401 (2017)
- Aziz, J. and Howl, R., "Classical theories of gravity produce entanglement," Nature 646, 49 (2025)
