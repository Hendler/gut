# Experiments

This file is the git-tracked summary ledger for important runs.

Bulky artifacts remain untracked:

- local log files in `/tmp`
- [results/diagnostics.svg](/Users/jonathan.hendler/personal/gut/results/diagnostics.svg)
- [results.tsv](/Users/jonathan.hendler/personal/gut/results.tsv)

## 2026-03-26

### Parameterized uniqueness sweep

- Commit: `eee38c3`
- Setup: Ordered family sweep over `g(r/sigma)` with increasing flexibility
- Families:
  - `erf(a x)`
  - `x^n / (c + x^n)`
  - `1 - exp(-(a x)^b)`
  - monotone piecewise-linear spline with 8 knots
- Best family: `erf_scaled`
- Best parameters: `a = 0.5`
- Best score: `0.000010344900`
- Runner-up: `rational_power` with `(n, c) = (4.0, 2.0)` and score `0.109019931626`
- Stretched exponential: `(a, b) = (0.6, 2.0)` and score `0.128425604273`
- Monotone spline: score `0.122852091552`
- Status: `keep`
- Note: The oracle strongly and uniquely selected `erf((r/sigma)/2)` within this sweep. Even the more flexible spline family did not approach the `erf` score, which suggests the current oracle has enough resolving power to distinguish the exact Gaussian-smearing shape among these candidate families.

### Dimensionless smearing-function baseline

- Commit: `eee38c3`
- Time budget: `5` seconds
- Unified score: `0.000010`
- Zero-formula score: `0.402024`
- Gravity error: `0.000000`
- Quantum error: `0.000000`
- Search rounds: `147`
- Formula: `V(r) = -G*mu*erf((r/sigma)/2)/r`
- Status: `keep`
- Note: Replaced the power-law basis search with a search over single transition functions `g(r/sigma)`. The oracle truth is now recovered directly as the simplest shared law, with only a tiny residual asymptotic penalty.

### Scientific weak-field baseline rerun

- Commit: `eee38c3`
- Time budget: `300` seconds
- Unified score: `0.065369`
- Zero-formula score: `0.478764`
- Gravity error: `0.000000`
- Quantum error: `0.065369`
- Search rounds: `992`
- Formula: `V(r) = +6.455642e-10*mu/r -7.122737e-10*mu/r_eff -3.649552e-10*mu*sigma^2/r_eff^3`
- Status: `baseline`
- Note: Repeated full-budget run on the Gaussian-smeared weak-field oracle. The result matched the previous best full-budget score, suggesting the current basis is stable and most remaining miss is on the quantum side.
