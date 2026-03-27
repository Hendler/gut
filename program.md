# autoresearch program for unified quantum-plus-gravity formula search

This repo is an `autoresearch`-style environment for symbolic physics search rather than language-model pretraining.

The target is not "find any low-loss expression."

The target is:

- one compact shared physical law
- that predicts gravity-side observables
- and quantum-side observables
- from the same underlying mathematical object

## Scientific Stance

Bias toward the simplest physically grounded search space first.

Do not begin from speculative frameworks such as:

- extra dimensions
- string-inspired oscillatory terms
- topology-first ansatze with no low-energy justification

Those ideas are not forbidden forever, but they are not the default starting point.

The default starting point should be:

- weak-field gravity
- proper time or action-based phase evolution
- effective-field-theory discipline
- symmetry, units, and limiting-behavior constraints

## Shared Object To Search For

Prefer a shared:

- action `S_theta`
- Hamiltonian `H_theta`
- or interaction potential `V_theta`

in that order of scientific priority.

Why:

- gravity observables can be derived from a shared interaction law
- quantum observables can be derived from phase evolution `exp(i S / hbar)` or `exp(-i H t / hbar)`
- this is a genuine unification target, not two separate predictors glued together

Time should be treated as physically meaningful proper-time evolution or invariant-duration input, not as an arbitrary feature hack.

## Two Layers

This project has two separate loops.

### Inner loop: local evaluator

The inner loop runs entirely on your machine with no API calls:

- `simulation.py` is the fixed oracle
- `train.py` evaluates one candidate search strategy against that oracle
- tests validate the oracle and experiment contracts

### Outer loop: LLM researcher

The outer loop makes this `autoresearch`-like:

- an LLM reads the repo instructions
- edits `train.py`
- runs tests
- runs `train.py`
- compares the score to the previous best
- keeps or discards the change
- repeats on a fixed cadence

If you want unattended autonomous iteration, this outer loop requires LLM access.

## Files To Read First

Before any experiment loop starts, read:

- `README.md`
- `REQUIREMENTS.md`
- `simulation.py`
- `train.py`
- `program.md`

`prepare.py` is legacy from the upstream repo and is out of scope here.

## Fixed vs Mutable Files

### Fixed files

Do not edit these during autonomous experimentation:

- `simulation.py`
- `REQUIREMENTS.md`
- `README.md`
- `tests/test_simulation.py`
- `tests/test_train.py`

### Mutable file

The outer loop edits:

- `train.py`

That is the main experimental surface.

## Core Metric

The canonical metric is:

- `unified_score`

Lower is better.

That score should reflect four priorities, in this order:

1. physical validity
2. held-out predictive accuracy
3. correct asymptotic limits
4. simplicity

For compatibility with the original notebook and log shape, `train.py` may also print:

- `val_bpb`

In this repo, `val_bpb` is only a compatibility alias for `unified_score`.

## What The Score Should Reward

The search should favor candidates that satisfy all of the following:

- one shared law predicts both output families
- dimensional consistency is respected
- symmetry under particle exchange is respected when appropriate
- weak-field / large-distance limits reduce to known behavior
- quantum evolution follows from the same law through phase accumulation
- performance holds on held-out regimes, not only in-distribution samples

The score should penalize:

- unit-inconsistent formulas
- formulas that violate known low-energy limits
- formulas that only fit one side well
- unnecessary complexity

## Test Gate

Before trusting any experiment result, the outer loop must run:

```bash
python3 -m unittest discover -s tests -v
```

If the tests fail, the iteration does not count.

## Standard Evaluation Command

Run one local evaluation with:

```bash
python3 train.py > run.log 2>&1
```

By default, `train.py` uses a 300-second time budget. For a shorter smoke test:

```bash
TIME_BUDGET_SECONDS=5 python3 train.py > run.log 2>&1
```

Then inspect:

```bash
grep "^unified_score:\|^val_bpb:\|^formula:" run.log
```

If that fails:

```bash
tail -n 50 run.log
```

## Five-Minute Outer Loop

To stay close to upstream `autoresearch`, each run of `train.py` should use a fixed time budget. The current default is 300 seconds.

Within that budget, `train.py` performs repeated formula-search rounds and reports the best result found before time expires.

The outer LLM loop then uses that result to decide what to change next.

The current default setup trains on the `train` regime and validates across:

- `heldout_compact`
- `heldout_decoherent`
- `heldout_wide`

## What To Explore In `train.py`

Useful directions include:

- richer shared-law basis libraries
- action-first or Hamiltonian-first parameterizations
- dimensional-analysis filtering
- sparsity and pruning heuristics
- better low-energy limit penalties
- symmetry constraints
- score weighting between gravity and quantum outputs
- dataset curricula and harder held-out cases
- formula distillation and reporting

## What Not To Do

Do not split the task into two unrelated black-box models and average them afterward.

Do not hard-code speculative theories as the default basis without low-energy motivation.

Do not assume that observing entanglement in a model automatically proves that gravity is quantum.

Do not modify `simulation.py` during the autonomous loop. If the oracle changes, scores stop being comparable.

Do not skip the tests.

## Diagnostics Contract

Each successful run should write a diagnostic artifact. The current baseline writes:

- `results/diagnostics.svg`

Removing diagnostics without replacing them with something equally useful is a regression.

## Results Logging

There are two result layers.

### Local untracked artifacts

Keep these untracked:

- `results/`
- `results.tsv`

These are convenient working artifacts.

### Git-tracked research history

Important experiment conclusions should also be written to a git-tracked lightweight ledger.

Recommended future file:

- `experiments.tsv` or `EXPERIMENTS.md`

Each kept run should record:

- date
- commit
- score
- formula
- keep or discard
- short description

Do not commit bulky plots by default.

## Experiment Loop

Once setup is complete, the outer LLM loop should do this:

1. Confirm the repo is clean and committed.
2. Record the current baseline if it has not been logged yet.
3. Modify only `train.py`.
4. Run the test suite.
5. Run `python3 train.py > run.log 2>&1`.
6. Extract the score and formula from the log.
7. Log the result in local `results.tsv`.
8. If the run is scientifically meaningful, also summarize it in the git-tracked experiment ledger.
9. If the score improves, keep the change.
10. If the score is equal or worse, discard the change.

## First Baseline Rule

The first recorded run should always be the current `train.py` with no code changes. After that, the outer loop can begin experimenting.
