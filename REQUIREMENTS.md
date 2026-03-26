# Requirements Plan For The Quantum Plus Gravity Autoresearch Loop

## Goal

Use the `autoresearch` workflow to search for a single explicit formula that can predict both:

- gravity-side outputs from a fixed oracle
- quantum-side outputs from the same fixed oracle

The target is not an immediate claim of a fundamental theory of quantum gravity. The first target is a shared effective formula that matches both kinds of oracle outputs within a controlled toy model.

## Core Architectural Decision

We will introduce a separate `simulation.py` and treat it as the unchanging oracle.

That means:

- `simulation.py` is fixed during autonomous experiments
- `train.py` is the file the agent changes during the search loop
- `program.md` is rewritten so the agent understands the new objective and loop

This follows the core `autoresearch` pattern: keep one fixed evaluation/oracle surface, and let the agent iterate on one main experimental file.

## File Roles

### `simulation.py`

This file is the fixed oracle.

Requirements:

- Must be deterministic and seedable.
- Must be lightweight enough to run many times during experimentation.
- Must not require external downloads or changing datasets.
- Must define the toy physical world we are trying to fit.
- Must expose both gravity and quantum outputs for the same underlying physical setup.
- Must remain unchanged during the autonomous loop.

### `train.py`

This file is the experimental search driver.

Requirements:

- Must import and use `simulation.py`.
- Must construct candidate third-formula models.
- Must optimize candidate formulas against oracle outputs.
- Must print a final scalar score so the `autoresearch` loop can compare runs.
- Must be the main file mutated by the agent during the search.

### `program.md`

This file becomes the research policy for the agent.

Requirements:

- Must instruct the agent that `simulation.py` is read-only.
- Must instruct the agent that `train.py` is the experimental surface.
- Must redefine success in terms of a unified physics score, not `val_bpb`.
- Must preserve the spirit of `autoresearch`: repeated small experiments, keep improvements, discard regressions.

## Oracle Design Requirements For `simulation.py`

The oracle should start with the simplest physically meaningful toy model already identified:

- two masses
- each mass in a two-position spatial superposition
- weak-field gravity
- quantum phase accumulation from branch-dependent gravitational interaction

### Minimum input parameters

Each sample should include a compact set of physical inputs such as:

- `m1`
- `m2`
- `base_distance`
- `delta1`
- `delta2`
- `interaction_time`

These inputs should fully define the branch geometry.

### Minimum gravity outputs

The oracle should provide gravity-side outputs derived from classical or weak-field formulas, such as:

- branch-dependent potential energies
- branch-dependent force magnitudes
- mean potential energy
- mean force magnitude

### Minimum quantum outputs

The oracle should provide quantum-side outputs derived from the same branch geometry, such as:

- branch-dependent phases
- recombined output probabilities
- an entanglement-related observable such as concurrence

### Practical oracle constraints

- Use only built-in Python plus already-available project dependencies.
- Prefer CPU-friendly computation.
- Use numerically stable units or scaled units so optimization is not dominated by tiny SI constants.
- Provide a simple importable API such as `oracle(sample)` and `make_dataset(num_samples, seed)`.
- Optionally provide a small CLI for manual inspection.

## What The Third Formula Must Be

The third formula must be a single shared mathematical object that predicts both output families.

The starting version should be:

- a shared potential `V_theta(x)`

From this shared formula:

- gravity predictions are derived from `V_theta(x)` or its spatial derivative
- quantum predictions are derived from phase accumulation using `V_theta(x)`

This is the simplest credible starting point because it gives one common mathematical core for both sides.

## What We Must Not Do

We should not start with two unrelated black-box models and average their outputs.

Disallowed starting pattern:

- one model only for gravity
- one separate model only for quantum
- a thin wrapper that pretends these are unified

That would not satisfy the actual research goal.

## Search Strategy Requirements In `train.py`

The search should begin with interpretable formula families.

Preferred starting approach:

- low-complexity basis terms
- learned coefficients
- sparsity pressure or term-pruning
- explicit formula reporting

Examples of candidate ingredients:

- inverse-distance terms
- polynomial corrections
- mixed interaction terms
- low-order rational terms

The output should stay interpretable enough that a discovered candidate can be written as a real formula.

## Optimization Objective

`train.py` should optimize one scalar score, lower is better.

Recommended structure:

```text
unified_score =
    gravity_error
  + alpha * quantum_error
  + beta * limit_penalty
  + gamma * complexity_penalty
```

Requirements:

- `gravity_error` must measure held-out mismatch on gravity outputs.
- `quantum_error` must measure held-out mismatch on quantum outputs.
- `limit_penalty` should reward known limiting behavior.
- `complexity_penalty` should prefer shorter and cleaner formulas.

## Known-Limit Requirements

The system should reward formulas that recover expected behavior in simple regimes.

Examples:

- large-distance behavior should resemble inverse-distance gravity
- weak-coupling behavior should give small quantum phase shifts
- symmetry under swapping the two masses should be respected when inputs are symmetric

## Interpretation Requirement

The final artifact we want is not just a low loss number.

We want:

- an explicit candidate formula
- its fitted coefficients
- its validation score
- a summary of where it succeeds and fails

If the search uses richer internal parameterizations, it should still distill down to a readable formula whenever possible.

## Research Phases

### Phase 1: Build the fixed oracle

Deliver `simulation.py` as the read-only source of truth for the toy system.

### Phase 2: Build a baseline unified fitter

Replace or rewrite `train.py` so it:

- samples from the oracle
- fits a shared formula
- reports `unified_score`
- reports the discovered formula in readable form

### Phase 3: Rewrite `program.md`

Update the loop instructions so the agent:

- treats `simulation.py` as fixed
- edits only `train.py`
- compares runs by `unified_score`
- logs experimental descriptions in the same iterative spirit as `autoresearch`

### Phase 4: Start autonomous search

Let the agent explore:

- basis libraries
- regularization strengths
- scaling and normalization choices
- candidate formula parameterizations
- pruning and simplification heuristics

## Success Criteria

We will consider the setup successful when:

- `simulation.py` is fixed and trustworthy as an oracle
- `train.py` can run end-to-end and report a stable `unified_score`
- the loop can compare experiments automatically
- the best run produces a readable shared formula, not just opaque weights
- the discovered formula improves over a simple baseline on both gravity and quantum targets

## Non-Goal Clarification

At this stage, the system is not proving a true theory of quantum gravity.

It is doing something narrower and more honest:

- searching for a single compact formula that jointly explains both classes of outputs in a controlled oracle world

That is the correct first step for using `autoresearch` productively on this problem.
