# autoresearch program for unified quantum-plus-gravity formula search

This repo is designed to mimic the structure of `karpathy/autoresearch`, but for symbolic physics search instead of language-model pretraining.

The target is a single compact formula that predicts both:

- gravity-side outputs from a fixed oracle
- quantum-side outputs from the same fixed oracle

## Two Layers

This project has two separate loops.

### Inner loop: local evaluator

The inner loop runs entirely on your machine with no API calls:

- `simulation.py` is the fixed oracle
- `train.py` evaluates one candidate search strategy against that oracle
- tests validate the oracle contract and experiment contract

### Outer loop: LLM researcher

The outer loop is the part that makes this truly `autoresearch`-like:

- an LLM reads the repo instructions
- edits `train.py`
- runs tests
- runs `train.py`
- compares the score to the previous best
- keeps or discards the change
- repeats on a fixed cadence

If you want unattended autonomous iteration, this outer loop requires LLM access. That can come from:

- an interactive Codex session like this one
- another coding agent
- an API-backed agent runner

The repo itself does not need an API key. The autonomous researcher does.

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

That score is now measured on held-out validation regimes, not just another sample from the training regime. Improvements should therefore reflect better generalization, not just better in-distribution fitting.

For compatibility with the original notebook and log shape, `train.py` also prints:

- `val_bpb`

In this repo, `val_bpb` is only a compatibility alias for `unified_score`.

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

By default, `train.py` uses a 300-second time budget. For a shorter smoke test, override it like this:

```bash
TIME_BUDGET_SECONDS=5 python3 train.py > run.log 2>&1
```

Then inspect:

```bash
grep "^unified_score:\|^val_bpb:\|^formula:" run.log
```

If that fails, inspect:

```bash
tail -n 50 run.log
```

## Five-Minute Outer Loop

To stay close to upstream `autoresearch`, each run of `train.py` should use a fixed time budget. The current default is 300 seconds.

Within that budget, `train.py` performs repeated formula-search rounds and reports the best result found before time expires.

The outer LLM loop then uses that result to decide what to change next.

The default setup trains on the `train` regime and validates across:

- `heldout_compact`
- `heldout_decoherent`
- `heldout_wide`

## Diagnostics Contract

Each successful run should write a diagnostic artifact. The current baseline writes:

- `results/diagnostics.svg`

Removing diagnostics without replacing them with something equally useful is a regression.

## Results Logging

Use `results.tsv` as an untracked local log. Keep the original 5-column schema:

```text
commit	val_bpb	memory_gb	status	description
```

Interpretation here:

1. `commit`: short git hash
2. `val_bpb`: the unified score
3. `memory_gb`: use `0.0` for now unless memory tracking is added later
4. `status`: `keep`, `discard`, or `crash`
5. `description`: short description of the experiment

## Experiment Loop

Once setup is complete, the outer LLM loop should do this forever:

1. Confirm the repo is clean and committed.
2. Record the current baseline if it has not been logged yet.
3. Modify only `train.py`.
4. Run the test suite.
5. Commit the experimental change.
6. Run `python3 train.py > run.log 2>&1`.
7. Extract the score and formula from the log.
8. Log the result in `results.tsv`.
9. If the score improves, keep the commit.
10. If the score is equal or worse, revert to the previous best commit.

## What To Explore In `train.py`

Useful directions include:

- richer basis libraries
- sparsity and pruning heuristics
- tie-breaking by simplicity
- better limit penalties
- score weighting between gravity and quantum outputs
- dataset curricula and harder held-out cases
- formula distillation and reporting

## What Not To Do

Do not split the task into two unrelated black-box models and average them afterward. That does not satisfy the actual goal.

Do not modify `simulation.py` during the autonomous loop. If the oracle changes, scores stop being comparable.

Do not skip the tests.

## First Baseline Rule

The first recorded run should always be the current `train.py` with no code changes. After that, the outer loop can begin experimenting.
