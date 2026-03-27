# Unified Quantum Plus Gravity Autoresearch

This repo is a small `autoresearch`-style environment for searching for a single compact formula that predicts both gravity-side and quantum-side behavior in a fixed weak-field simulator.

The current oracle world is:

- two masses
- each mass in a two-position spatial superposition
- weak-field gravity in SI units
- Gaussian wavepacket smearing of the gravitational interaction
- branch-dependent quantum phase accumulation from the interaction energy
- two-qubit recombination probabilities after local Hadamards
- finite coherence-length washout at readout

The point is not to claim a true theory of quantum gravity right away. The point is to build an honest search loop for a shared effective formula inside a controlled oracle world.

The project has now reached an important first milestone:

- the search no longer uses a power-law correction soup
- it searches directly over a dimensionless transition function `g(r/sigma)`
- and it recovers the oracle truth `erf((r/sigma)/2)` as the best shared law

## What Is In This Repo

- [simulation.py](/Users/jonathan.hendler/personal/gut/simulation.py): fixed oracle for the weak-field physics model
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

For branch `ab`, the oracle starts from a branch-dependent interaction energy:

```text
U_ab = -G m1 m2 erf(r_ab / (2 sigma)) / r_ab
```

This is the Newtonian interaction averaged over two equal-width Gaussian mass packets separated by `r_ab`.

The corresponding branch phase is:

```text
phi_ab = -U_ab t / hbar
```

The oracle then derives:

- gravity outputs such as branch forces, mean potential, and force spread
- quantum outputs such as recombined probabilities, concurrence, and visibility

The current oracle is more scientific than the original toy. It now includes:

- physical SI constants
- Gaussian spatial smearing instead of hand-built `r_eff` corrections
- exact diagonal branch evolution in the four-branch basis
- a coherence-length overlap factor for readout visibility
- explicit held-out validation regimes for generalization

The current search in [train.py](/Users/jonathan.hendler/personal/gut/train.py) is organized around the physically meaningful form:

```text
V(r) = -G*mu*g(r/sigma)/r
```

The question is not "which sum of correction terms best fits the data?"

The question is:

- what is the simplest transition function `g(x)`
- that connects the short-distance smeared regime to the large-distance Newtonian regime
- while predicting both gravity and quantum observables

## Current Baseline

The current baseline search in [train.py](/Users/jonathan.hendler/personal/gut/train.py) fits on the `train` regime and validates on three held-out regimes:

- `heldout_compact`
- `heldout_decoherent`
- `heldout_wide`

With a 5-second smoke-test budget, the recovered shared law is:

```text
V(r) = -G*mu*erf((r/sigma)/2)/r
```

with output like:

```text
unified_score: 0.000010
val_bpb:       0.000010
gravity_error: 0.000000
quantum_error: 0.000000
```

The tiny remaining score is just the asymptotic limit penalty, not a miss on the oracle outputs.

The parameterized uniqueness sweep now also shows that the oracle strongly prefers the Gaussian-smearing answer:

- `erf(a x)` recovers `a = 0.5`
- rational, stretched-exponential, and spline families score much worse

That means the current oracle is strong enough to distinguish the exact smearing law among the candidate families we tested.

Here `val_bpb` is only a compatibility alias for the `autoresearch`-style log format. It does not mean bits per byte in this repo.

## Next Experiment

The next meaningful step is not just to make the oracle "harder."

The next meaningful step is to make it harder in a way that carries real physics content:

- keep the recovered smearing law `g(x) = erf(x/2)`
- enrich the oracle with weak-field EFT corrections
- then ask whether symbolic search can rediscover those corrections from data

The target correction structure is:

```text
V(r) = -G m1 m2 / r * erf(r/(2*sigma)) *
       [1 + 3G(m1+m2)/(r c^2) + (41/(10*pi))*G*hbar/(r^2 c^3)]
```

This separates into:

- Newtonian term
- post-Newtonian classical GR correction
- Donoghue quantum-gravity EFT correction

The methodological question is:

- can automated symbolic search recover the known coefficients `3` and `41/(10*pi)` from simulated weak-field data?

That would not be new physics, but it would be an interesting and honest computational result.

The planned workflow is:

1. Add the post-Newtonian and Donoghue corrections to [simulation.py](/Users/jonathan.hendler/personal/gut/simulation.py).
2. Freeze the smearing law at `erf((r/sigma)/2)`.
3. Search only over dimensionally consistent residual correction terms.
4. Perform a second uniqueness sweep over the correction families.
5. Check whether the recovered coefficients agree with the known EFT values.

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

To view the graph in the desktop app, open [results/diagnostics.svg](/Users/jonathan.hendler/personal/gut/results/diagnostics.svg). To inspect run-to-run progress after multiple iterations, use [results.tsv](/Users/jonathan.hendler/personal/gut/results.tsv).

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

- a fixed weak-field BMV-style oracle
- a shared-formula scoring problem
- a testable local evaluator
- an outer-loop policy that an LLM can use to perform repeated research iterations

That is the correct starting point for an `autoresearch`-style search here.
