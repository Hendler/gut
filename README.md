# Unified Quantum Plus Gravity Autoresearch

This repo is a small `autoresearch`-style environment for searching for a single compact formula that predicts both gravity-side and quantum-side behavior in a fixed toy model.

The current toy world is:

- two masses
- each mass in a two-position spatial superposition
- weak-field gravity
- quantum phase accumulation from branch-dependent gravitational interaction
- finite wavepacket size
- coherence-dependent visibility loss

The point is not to claim a true theory of quantum gravity right away. The point is to build an honest search loop for a shared effective formula inside a controlled oracle world.

## What Is In This Repo

- [simulation.py](/Users/jonathan.hendler/personal/gut/simulation.py): fixed oracle for the toy physics model
- [train.py](/Users/jonathan.hendler/personal/gut/train.py): local experiment runner that scores candidate unified formulas
- [program.md](/Users/jonathan.hendler/personal/gut/program.md): instructions for the outer `autoresearch`-style LLM loop
- [REQUIREMENTS.md](/Users/jonathan.hendler/personal/gut/REQUIREMENTS.md): design requirements and research framing
- [tests/test_simulation.py](/Users/jonathan.hendler/personal/gut/tests/test_simulation.py): oracle contract tests
- [tests/test_train.py](/Users/jonathan.hendler/personal/gut/tests/test_train.py): experiment contract tests

## Two Separate Things

There are two different layers here.

### 1. Local evaluator

This runs entirely on your machine:

- `simulation.py` generates the fixed oracle outputs
- `train.py` searches over candidate formulas and prints a scalar score
- tests check that the contract still holds

No API key is required for this part.

### 2. Outer autoresearch loop

This is the real `autoresearch` part:

- an LLM reads the repo instructions
- edits `train.py`
- runs the tests
- runs `train.py`
- compares the new score to the old score
- keeps or discards the change
- repeats

This part does require LLM access if you want it to run unattended.

That LLM access can come from:

- a live Codex session like this one
- another coding agent
- an API-backed agent runner

So the short rule is:

- local scoring and testing: no API key
- autonomous LLM researcher: yes, some LLM access is needed

## Physics Model

Each mass has two possible positions:

- `|L>`
- `|R>`

The branch basis is:

- `|LL>`
- `|LR>`
- `|RL>`
- `|RR>`

For branch `ab`, the oracle starts from a branch-dependent gravitational potential of the form:

```text
U_ab ~ -G m1 m2 / r_eff + weak-field corrections + geometry/coherence corrections
```

and the corresponding quantum phase:

```text
phi_ab = -U_ab t / hbar
```

The toy oracle then derives:

- gravity outputs such as branch forces and mean potential
- quantum outputs such as recombined probabilities, concurrence, and visibility

The current oracle is harder than the original Newtonian toy. It now includes:

- softened effective distance from finite wavepacket size
- a weak-field post-Newtonian correction term
- a geometry/coherence correction that is not in the current search basis
- quantum visibility loss driven by branch force spread and coherence length
- explicit held-out validation regimes for generalization

## Current Baseline

The current baseline search in [train.py](/Users/jonathan.hendler/personal/gut/train.py) now fits on the `train` regime and validates on three held-out regimes:

- `heldout_compact`
- `heldout_decoherent`
- `heldout_wide`

With a 5-second smoke-test budget, the best current approximate shared potential is:

```text
V(r) = -0.058137*mu/r -0.945076*mu/r_eff -0.036002*mu/r_eff^3
```

with output like:

```text
unified_score: 0.002258
val_bpb:       0.002258
gravity_error: 0.001710
quantum_error: 0.000547
```

That nonzero score is intentional: the oracle is now richer than the current basis, so the search must approximate and generalize instead of recovering the exact hidden law.

Here `val_bpb` is only a compatibility alias for the `autoresearch`-style log format. It does not mean bits per byte in this repo.

## How To Use Locally

### Run the tests

```bash
python3 -m unittest discover -s tests -v
```

### Preview the oracle

```bash
python3 simulation.py --samples 4 --seed 0
```

### Run one local evaluation

```bash
python3 train.py
```

By default this now runs a 300-second bounded search, closer to the original `autoresearch` pattern.

For a quick smoke test, use:

```bash
TIME_BUDGET_SECONDS=5 python3 train.py
```

This prints the main metrics, including how many search rounds fit inside the budget, and writes a diagnostic plot to:

- [results/diagnostics.svg](/Users/jonathan.hendler/personal/gut/results/diagnostics.svg)

The diagnostic plot is drawn from one held-out regime, while the printed `unified_score` averages across all configured held-out validation regimes.

## How To Use With An LLM In The Loop

If you want the true `autoresearch` pattern, use the repo like this:

1. Make sure the repo is clean and committed.
2. Initialize `results.tsv` if it does not exist yet.
3. Give the agent [program.md](/Users/jonathan.hendler/personal/gut/program.md) as the outer-loop policy.
4. Keep [simulation.py](/Users/jonathan.hendler/personal/gut/simulation.py) fixed.
5. Let the agent modify only [train.py](/Users/jonathan.hendler/personal/gut/train.py).
6. Require the agent to run the test suite before trusting an experiment.
7. Require the agent to keep or discard each experiment based on `unified_score`.

The baseline setup now mirrors upstream more closely:

- each run of [train.py](/Users/jonathan.hendler/personal/gut/train.py) uses a fixed time budget
- inside that budget it performs repeated search rounds
- then the outer LLM loop decides how to modify `train.py` for the next run

The default time budget is 300 seconds. You can override it for smoke tests with `TIME_BUDGET_SECONDS`.

## Suggested Results Log

Create a local `results.tsv` with this header:

```text
commit	val_bpb	memory_gb	status	description
```

Use the columns as:

- `commit`: short git hash
- `val_bpb`: unified score
- `memory_gb`: `0.0` for now
- `status`: `keep`, `discard`, or `crash`
- `description`: short summary of the change

## Current Limits

This repo does not yet prove a true theory of quantum gravity.

What it does provide is:

- a fixed oracle world
- a shared-formula scoring problem
- a testable local evaluator
- an outer-loop policy that an LLM can use to perform repeated research iterations

That is the correct starting point for an `autoresearch`-style search here.
