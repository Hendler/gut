# autoresearch for unified quantum-plus-gravity formula search

This repository is no longer a language-model training project.

The goal is to use the `autoresearch` loop to search for a single explicit formula that can predict both gravity-side and quantum-side outputs from a fixed toy oracle.

## Read These Files First

Before starting any experiment loop, read:

- `README.md`
- `REQUIREMENTS.md`
- `simulation.py`
- `train.py`
- `program.md`

`prepare.py` is legacy from the original upstream repo and is out of scope for this project.

## Fixed vs Mutable Files

### Fixed files

These are the harness and should not be edited during autonomous experimentation:

- `simulation.py`
- `REQUIREMENTS.md`
- `README.md`
- `tests/test_simulation.py`
- `tests/test_train.py`

### Mutable file

During the experiment loop, the agent edits:

- `train.py`

This is the main search surface.

## Core Objective

The objective is to discover a compact shared formula that predicts both:

- gravity-side oracle outputs
- quantum-side oracle outputs

The canonical score is:

- `unified_score`

Lower is better.

For compatibility with the original `autoresearch` output format, `train.py` also prints:

- `val_bpb`

In this repo, `val_bpb` is just an alias for `unified_score`. It does not mean bits per byte.

## What Counts As A Good Experiment

A good experiment improves one or more of:

- held-out gravity error
- held-out quantum error
- asymptotic-limit behavior
- formula simplicity

The final artifact should remain interpretable. Prefer experiments that improve the score without making the formula search much uglier.

## Test-Driven Requirement

This repo should stay testable.

Before running an experiment, run:

```bash
python3 -m unittest discover -s tests -v
```

If the tests fail, fix the issue before trusting any experiment result.

## Standard Run Command

Run the experiment like this:

```bash
python3 train.py > run.log 2>&1
```

Then extract the important lines:

```bash
grep "^unified_score:\|^val_bpb:\|^formula:" run.log
```

If the grep output is empty, the run failed. Inspect:

```bash
tail -n 50 run.log
```

## Diagnostics

Each successful run should generate a diagnostics plot file. Right now the baseline implementation writes:

- `results/diagnostics.svg`

The plot is part of the contract. If a run stops generating diagnostics, treat that as a regression unless there is a very good reason.

## Results Logging

Keep using `results.tsv` as an untracked local experiment log.

For compatibility with the original notebook shape, keep the existing 5-column schema:

```text
commit	val_bpb	memory_gb	status	description
```

Interpretation in this repo:

1. `commit`: short git hash
2. `val_bpb`: the unified score
3. `memory_gb`: use `0.0` for now unless memory tracking is added later
4. `status`: `keep`, `discard`, or `crash`
5. `description`: short summary of the experiment

## The Experiment Loop

Once setup is complete, loop like this:

1. Check git state and current best commit.
2. Edit `train.py` with one experimental change.
3. Run the test suite.
4. Commit the change.
5. Run `python3 train.py > run.log 2>&1`.
6. Read the score from `run.log`.
7. If the run crashes, inspect the stack trace and decide whether to fix or discard.
8. Log the outcome in `results.tsv`.
9. If the unified score improves, keep the commit.
10. If it does not improve, revert to the previous best state.

## What To Explore In `train.py`

Useful experiment directions include:

- basis-term libraries
- complexity penalties
- normalization and scaling choices
- sparsity or pruning heuristics
- coefficient fitting methods
- limit penalties
- score weighting between gravity and quantum outputs

## What Not To Do

Do not turn the problem into two separate unrelated models and a late-stage average. The whole point is to search for one shared mathematical object.

Do not edit the oracle during autonomous experiments. If the oracle changes, comparisons across runs become meaningless.

Do not remove the diagnostics output without replacing it with something equally useful.

## First Baseline Rule

The first experimental run should always be the current baseline `train.py` as-is, with no changes. Record that score before starting further exploration.
