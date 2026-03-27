# Requirements Plan For The Quantum Plus Gravity Autoresearch Loop

## Goal

Use the `autoresearch` workflow to search for a single explicit shared law that predicts both:

- gravity-side outputs from a fixed oracle
- quantum-side outputs from the same fixed oracle

The target is not an immediate claim of a final theory of quantum gravity.

The first real target is:

- a compact shared effective law
- with correct units and limits
- that predicts both output families
- and generalizes across held-out regimes

This first target has effectively been met for the Gaussian-smearing oracle:

- the search now works in terms of a dimensionless transition function `g(r/sigma)`
- the oracle truth `erf((r/sigma)/2)` is recovered
- and the uniqueness sweep strongly prefers that function over the tested alternatives

The next target should therefore be more ambitious and more physically meaningful:

- recover the leading weak-field EFT corrections on top of the known smearing law

## Core Architectural Decision

We use a separate `simulation.py` as the fixed oracle.

That means:

- `simulation.py` is fixed during autonomous experiments
- `train.py` is the file the agent changes during the search loop
- `program.md` defines the research policy and evaluation discipline

This follows the core `autoresearch` pattern: keep one fixed evaluation surface, and let the agent iterate on one main experimental file.

## Scientific Direction

The repo should default toward the simplest scientifically grounded unification path first.

Preferred framing:

- weak-field gravity
- proper time as a physical quantity, not a heuristic feature
- action-based, Hamiltonian-based, or potential-based shared laws
- effective-field-theory discipline at low energies
- dimensional consistency
- known symmetry and asymptotic constraints

Not preferred as the default starting point:

- string-inspired speculative ansatze
- extra-dimension oscillation terms
- topology-first search spaces with no low-energy justification

These ideas are not banned forever. They are just not the default basis family.

## Why This Direction

The most credible route to "one formula for both" is:

- derive gravity observables from a shared interaction law
- derive quantum observables from the same law through phase evolution

Examples:

- `exp(i S / hbar)`
- `exp(-i H t / hbar)`

This keeps the search honest. We are not looking for two separate predictors glued together after the fact.

## What The Shared Law Must Be

The search should prefer a single shared mathematical object, ideally one of:

1. an action `S_theta`
2. a Hamiltonian `H_theta`
3. a potential `V_theta`

The current code starts with the simplest version, a shared interaction law used for both:

- gravity predictions through potential and force
- quantum predictions through branch phase

This is the minimum viable unification target.

## File Roles

### `simulation.py`

This file is the fixed oracle.

Requirements:

- deterministic and seedable
- lightweight enough for repeated local experiments
- no external downloads or changing datasets
- one shared physical setup that exposes both gravity and quantum outputs
- unchanged during the autonomous loop

### `train.py`

This file is the experimental search driver.

Requirements:

- import and use `simulation.py`
- construct candidate shared-law models
- optimize candidates against oracle outputs
- print a final scalar score
- remain interpretable enough to report an explicit formula

### `program.md`

This file is the research policy for the agent.

Requirements:

- tell the agent that `simulation.py` is read-only
- tell the agent that `train.py` is the experimental surface
- define success in terms of unified predictive performance and physical validity
- preserve the repeated-small-experiment spirit of `autoresearch`

## Oracle Requirements For `simulation.py`

The oracle should remain the simplest physically meaningful low-energy model we can defend.

Current preferred setup:

- two masses
- each mass in a two-position spatial superposition
- weak-field gravity
- Gaussian spatial smearing of the interaction
- quantum phase accumulation from the branch-dependent interaction energy
- coherence-limited readout visibility

Near-term enriched setup:

- keep the Gaussian smearing factor fixed
- add the classical post-Newtonian correction
- add the leading Donoghue quantum-gravity EFT correction

### Minimum input parameters

Each sample should include a compact set of physical inputs such as:

- `m1`
- `m2`
- `base_distance`
- `delta1`
- `delta2`
- `interaction_time`
- `wavepacket_width`
- `coherence_length`

### Minimum gravity outputs

The oracle should provide gravity-side outputs such as:

- branch-dependent interaction energies
- branch-dependent force magnitudes
- mean potential energy
- mean force magnitude
- force spread

### Minimum quantum outputs

The oracle should provide quantum-side outputs such as:

- branch-dependent phases
- recombined output probabilities
- concurrence or another entanglement-related observable
- visibility

### Practical oracle constraints

- use built-in Python plus already-available project dependencies unless a deliberate dependency upgrade is approved
- prefer CPU-friendly computation
- maintain numerical stability
- expose a simple importable API such as `oracle(sample)` and `make_dataset(...)`

## Search Strategy Requirements In `train.py`

The search proceeds hierarchically:

1. **Smearing layer** (solved): `g(r/sigma) = erf((r/sigma)/2)`
2. **Correction layer** (solved with known basis): `h(r) = 1 + Σ c_k f_k` where `f_k` are dimensionally-consistent monomials
3. **Blind correction layer** (next): same structure, but `f_k` generated programmatically from dimensional analysis without oracle knowledge

Each layer uses explicit formula reporting and held-out validation.

The blind search enumerates dimensionless monomials `G^a * hbar^b * c^d * M^e * r^f * sigma^g` subject to dimensional constraints, then fits coefficients via ridge regression over all subsets up to a maximum size.

## Optimization Objective

`train.py` should optimize one scalar score, lower is better.

Recommended structure:

```text
unified_score =
    physical_validity_penalty
  + gravity_error
  + alpha * quantum_error
  + beta * limit_penalty
  + gamma * complexity_penalty
```

Requirements:

- `gravity_error` must measure held-out mismatch on gravity outputs
- `quantum_error` must measure held-out mismatch on quantum outputs
- `physical_validity_penalty` should heavily punish unit mistakes or broken symmetry requirements
- `limit_penalty` should reward known limiting behavior
- `complexity_penalty` should prefer shorter and cleaner formulas

## Known-Limit Requirements

The system should reward formulas that recover expected behavior in simple regimes.

Examples:

- large-distance behavior should approach inverse-distance gravity
- weak coupling should give small quantum phase shifts
- symmetric inputs should respect particle-exchange symmetry
- time dependence should enter in a physically interpretable way

For the enriched EFT oracle, formulas should also recover:

- the classical post-Newtonian scaling `G M / (r c^2)`
- the leading quantum-gravity EFT scaling `G hbar / (r^2 c^3)`

## Interpretation Requirement

The final artifact is not just a low loss number.

We want:

- an explicit candidate formula
- fitted coefficients
- validation score
- a summary of where it succeeds and fails

If the search uses richer internal parameterizations, it should still distill down to a readable formula whenever possible.

## Research Cautions

The agent should explicitly avoid overclaiming.

Important caution:

- the older BMV-style intuition that gravity-mediated entanglement would straightforwardly prove gravity is quantum is no longer safe as a hard-coded assumption

A later result argues that classical gravity can also generate entanglement in an appropriate QFT treatment of matter. This means the project goal should stay predictive and comparative:

- find the simplest shared law that matches the oracle
- compare candidate effective theories
- do not claim that one signature alone settles the ontology of gravity

Second caution:

- recovering the Donoghue coefficients from a simulated oracle would be a methodological validation, not evidence that the corrections have been experimentally measured

## Results And Git History

Current situation:

- `results.tsv` is a local untracked working log
- `results/` is an untracked artifact directory

That is fine for fast iteration, but important conclusions should also be saved in git through a lightweight tracked ledger.

Git-tracked ledger:

- `EXPERIMENTS.md`

Each kept run should record:

- date
- commit
- score
- formula
- keep or discard
- short note

Bulky plots should remain untracked unless there is a specific reason to preserve one.

## Research Phases

### Phase 1: Smearing-function recovery (complete)

- Searched over dimensionless transition functions `g(r/sigma)`
- Recovered `erf((r/sigma)/2)` as the unique best law
- Uniqueness sweep confirmed: erf beats rational, stretched-exponential, and spline families by large margins

### Phase 2: EFT correction recovery with known basis (complete)

- Oracle enriched with post-Newtonian and Donoghue corrections
- Analytic force derivative preserves correction structure
- Amplified config (G=1, hbar=0.5, c=2) makes corrections resolvable
- 5-term dimensionally-consistent basis (2 real + 3 spurious)
- Recovered coefficients: `pn_classical = 3.0`, `quantum_eft = 41/(10*pi)`, spurious = 0
- CLI: `python3 train.py --mode eft`

### Phase 3: Blind recovery (complete)

Remove knowledge of which terms are in the oracle:

1. Enumerate all dimensionless monomials in `{G, hbar, c, M, r, sigma}` up to second order in `G` and `hbar`
2. Exhaustive subset search up to size 3 across the blind library
3. Fit coefficients via regression, score on held-out EFT-sensitive regimes
4. Check whether the search independently discovers `{G*M/(r*c^2), G*hbar/(r^2*c^3)}` with coefficients 3.0 and 1.305

Implemented result:

- Blind library size: 29 terms
- Exhaustive subsets up to size 3: 4089
- Winning subset: `{G*M/(r*c^2), G*hbar/(r^2*c^3)}`
- Recovered coefficients: `3.0` and `41/(10*pi)`

CLI: `python3 train.py --mode blind`

### Phase 4: Correction degeneracy analysis (complete)

Apply the same uniqueness methodology used for `g(r/sigma)` to the correction sector:

- Rank the full blind subset space instead of only keeping the winner
- Canonicalize away exactly zero coefficients so padded supersets do not masquerade as distinct formulas
- Measure the gap between rank 1 and rank 2

Current result:

- Rank 1: `{G*M/(r*c^2), G*hbar/(r^2*c^3)}` with score `0.000000`
- Rank 2: `{G*M/(r*c^2), G*M*sigma^3/(r^4*c^2), G*hbar*sigma/(r^3*c^3)}` with score `0.000396`
- Runner-up margin: `0.000396`

Interpretation:

- The oracle does uniquely select the Donoghue pair within the blind search space
- The uniqueness margin is real but moderate, so richer regimes would still be valuable if we want a sharper separation

CLI: `python3 train.py --mode degeneracy`

## Success Criteria

We will consider the setup successful when:

- `simulation.py` is fixed and trustworthy as an oracle
- `train.py` runs end to end and reports a stable `unified_score`
- the loop compares experiments automatically
- the best run produces a readable shared law, not opaque weights
- the discovered law improves over a simple baseline on both gravity and quantum targets

For the next stage, success also includes:

- recovery of the correct correction structure on top of the smearing law
- coefficients close to the known EFT values within oracle precision
- uniqueness or quantified degeneracy of that recovery

## Non-Goal Clarification

At this stage, the system is not proving a final theory of quantum gravity.

It is doing something narrower and more honest:

- searching for a single compact law that jointly explains both classes of outputs in a controlled oracle world

In the next stage, the most honest headline is:

- "automated symbolic search recovers known weak-field EFT corrections from a simulated oracle"

not:

- "we discovered new quantum gravity physics"

That is the correct first step for using `autoresearch` productively on this problem.

## References

- Donoghue, J. F., “General relativity as an effective field theory: The leading quantum corrections,” *Physical Review D* 50, 3874-3888 (1994). DOI: [10.1103/PhysRevD.50.3874](https://doi.org/10.1103/PhysRevD.50.3874)
- Zych, M., Costa, F., Pikovski, I., and Brukner, C., “Quantum interferometric visibility as a witness of general relativistic proper time,” *Nature Communications* 2, 505 (2011). DOI: [10.1038/ncomms1498](https://doi.org/10.1038/ncomms1498)
- Bose, S. et al., “Spin Entanglement Witness for Quantum Gravity,” *Physical Review Letters* 119, 240401 (2017). DOI: [10.1103/PhysRevLett.119.240401](https://doi.org/10.1103/PhysRevLett.119.240401)
- Marletto, C. and Vedral, V., “Gravitationally Induced Entanglement between Two Massive Particles Is Sufficient Evidence of Quantum Effects in Gravity,” *Physical Review Letters* 119, 240402 (2017). DOI: [10.1103/PhysRevLett.119.240402](https://doi.org/10.1103/PhysRevLett.119.240402)
- Aziz, J. and Howl, R., “Classical theories of gravity produce entanglement,” *Nature* 646, 49-53 (published October 22, 2025). DOI: [10.1038/s41586-025-09595-7](https://doi.org/10.1038/s41586-025-09595-7)
