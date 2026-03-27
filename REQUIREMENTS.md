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

The search should begin with interpretable formula families.

Preferred starting approach:

- phase 1: search over dimensionless transition functions `g(r/sigma)`
- phase 2: hold the recovered `g(r/sigma)` fixed
- phase 3: search over dimensionally consistent correction terms on top of that law
- explicit formula reporting
- held-out validation

Examples of candidate ingredients:

- dimensionless smearing functions
- post-Newtonian correction terms
- EFT correction monomials built from `G`, `hbar`, `c`, masses, and `r`
- action-inspired or Hamiltonian-inspired parameterizations

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

### Phase 1: Fixed oracle

Maintain `simulation.py` as the read-only source of truth for the toy system.

### Phase 2: Baseline unified fitter

Use `train.py` to:

- sample from the oracle
- fit a shared law
- report `unified_score`
- report the discovered formula in readable form

### Phase 3: Research-policy refinement

Use `program.md` to enforce:

- read-only oracle
- mutable `train.py`
- physically disciplined scoring
- iterative keep or discard behavior

### Phase 4: Autonomous search

Let the agent explore:

- basis libraries
- regularization and scaling
- dimensional-analysis filters
- action or Hamiltonian parameterizations
- pruning and simplification heuristics

### Phase 5: Enriched EFT oracle

The next oracle revision should be physically motivated, not arbitrary.

Planned correction structure:

```text
V(r) =
  -G m1 m2 / r
  * erf(r / (2*sigma))
  * [1 + 3G(m1+m2)/(r c^2) + (41/(10*pi))*G*hbar/(r^2 c^3)]
```

Interpretation:

1. Newtonian term
2. classical post-Newtonian correction
3. leading Donoghue quantum-gravity EFT correction

This is the next oracle worth spending substantial compute on.

### Phase 6: Hierarchical blind recovery

Once the enriched oracle exists, the search should proceed hierarchically:

1. Fix the already recovered smearing law `g(x) = erf(x/2)`.
2. Search for residual correction families on top of it.
3. Restrict the search to dimensionally consistent terms.
4. Fit coefficients without hard-coding the target values.
5. Measure whether the recovered coefficients approach:
   - `3`
   - `41/(10*pi)`

### Phase 7: Correction degeneracy analysis

The same uniqueness question used for `g(r/sigma)` should then be applied to the correction sector.

Key question:

- are the EFT correction terms uniquely recoverable from the oracle, or do multiple correction families score equally well?

If degeneracy appears, the oracle must be enriched further through:

- more held-out regimes
- shorter-distance probes
- higher phase sensitivity
- larger datasets and repeated draws

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
