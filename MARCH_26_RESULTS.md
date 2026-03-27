# March 26 Results

## Summary

Within a 29-term blind dimensional-analysis search over an amplified weak-field oracle, automated symbolic regression robustly recovered the Donoghue quantum-gravity EFT correction structure:

```text
V(r) = -G*mu*erf((r/sigma)/2)/r * [1 + 3*G*M/(r*c^2) + (41/(10*pi))*G*hbar/(r^2*c^3)]
```

The recovery was stable across:

- random seed changes
- larger datasets
- search-space enlargement from 4,089 subsets to 27,840 subsets

The winner remained the same in every informative run:

- `G*M/(r*c^2)`
- `G*hbar/(r^2*c^3)`

with recovered coefficients:

- `3.000000`
- `1.305071`

## What Was Tested

The oracle in `simulation.py` used:

- Gaussian smearing `erf(r / (2*sigma))`
- leading post-Newtonian correction
- Donoghue leading quantum EFT correction

The search in `train.py` used:

- an amplified methodological config: `G = 1`, `hbar = 0.5`, `c = 2`
- a blind dimensional enumerator over monomials in `{G, hbar, c, M, r, sigma}`
- exhaustive subset search
- held-out validation on `eft_sensitive_compact` and `eft_sensitive_wide`

## Main Results

### Phase 2: Known-Basis Recovery

Curated 5-term basis:

- recovered `pn_classical = 3.0`
- recovered `quantum_eft = 41/(10*pi)`
- suppressed spurious terms to `0`

This established that the fitter could recover the right coefficients when the true terms were present in the candidate basis.

### Phase 3: Blind Recovery

Blind library size:

- `29` dimensionally consistent candidate terms

Blind subset search:

- up to subset size `3`
- total subsets searched: `4089`

Winner:

- `{G*M/(r*c^2), G*hbar/(r^2*c^3)}`

Recovered formula:

```text
V(r) = -G*mu*erf((r/sigma)/2)/r * [1 +3.000000*G*M/(r*c^2) +1.305071*G*hbar/(r^2*c^3)]
```

## Research Loop Runs

### 1. Baseline Degeneracy Run

Configuration:

- `num_train = 160`
- `num_val = 96`
- `seed = 41`
- `max_subset_size = 3`

Result:

- winner: `{G*M/(r*c^2), G*hbar/(r^2*c^3)}`
- runner-up margin: `0.000402`

Top competitors:

1. `{G*M/(r*c^2), G*hbar/(r^2*c^3)}` with score `0.000000`
2. `{G*M/(r*c^2), G*M*sigma^3/(r^4*c^2), G*hbar*sigma/(r^3*c^3)}` with score `0.000402`
3. `{G*M/(r*c^2), G*M*sigma^4/(r^5*c^2), G*hbar*sigma/(r^3*c^3)}` with score `0.000430`

### 2. Seed Variation

Configuration:

- `num_train = 160`
- `num_val = 96`
- `seed = 53`
- `max_subset_size = 3`

Result:

- winner unchanged: `{G*M/(r*c^2), G*hbar/(r^2*c^3)}`
- runner-up margin increased to `0.000519`

Interpretation:

- the winner was invariant
- the runner-up changed shape

This is the clearest evidence that the Donoghue pair is a real signal while the spurious alternatives are dataset-sensitive approximations.

### 3. Larger Dataset

Configuration:

- `num_train = 320`
- `num_val = 192`
- `seed = 41`
- `max_subset_size = 3`

Result:

- winner unchanged: `{G*M/(r*c^2), G*hbar/(r^2*c^3)}`
- no genuinely better competitor appeared

Interpretation:

- increasing data did not reveal a new rival law
- near-duplicates collapsed onto the same Donoghue pair after canonicalization

### 4. Enlarged Search Space

Configuration:

- `num_train = 160`
- `num_val = 96`
- `seed = 41`
- `max_subset_size = 4`

Search size:

- total subsets searched: `27,840`

Result:

- winner unchanged: `{G*M/(r*c^2), G*hbar/(r^2*c^3)}`
- cleaned runner-up margin: `0.000302`

Top competitors after coefficient pruning and formula-text deduplication:

1. `{G*M/(r*c^2), G*hbar/(r^2*c^3)}` with score `0.000000`
2. `{hbar^2*sigma^4/(M^2*r^6*c^2), G*M/(r*c^2), G*M*sigma^3/(r^4*c^2), G*hbar*sigma/(r^3*c^3)}` with score `0.000302`
3. `{G*M/(r*c^2), G*M*sigma/(r^2*c^2), G*M*sigma^4/(r^5*c^2), G*hbar*sigma/(r^3*c^3)}` with score `0.000312`
4. `{hbar^2*sigma^3/(M^2*r^5*c^2), G*M/(r*c^2), G*M*sigma^3/(r^4*c^2), G*hbar*sigma/(r^3*c^3)}` with score `0.000314`
5. `{G*M/(r*c^2), G*M*sigma^2/(r^3*c^2), G*hbar*sigma/(r^3*c^3), G*hbar*sigma^2/(r^4*c^3)}` with score `0.000317`

Interpretation:

- search-space enlargement did not improve on the 2-term Donoghue pair
- the best competing formulas remain modestly worse

## Cleanup Performed

To make the degeneracy ranking honest:

- coefficient pruning was changed from a tiny absolute cutoff to a scale-aware tolerance
- duplicate formulas were deduplicated by rendered formula text

This removed phantom entries that differed only by numerically negligible coefficients.

## Honest Conclusion

Within this oracle and search space, the result is:

> Automated symbolic regression uniquely and robustly recovers the Donoghue quantum gravity EFT correction structure from an amplified weak-field oracle.

What this is:

- a methodological validation
- evidence that the search framework can recover the right low-energy EFT structure from simulated data

What this is not:

- new physics
- an experimental measurement
- evidence that the Donoghue correction has been observed in the real world

## Most Important Takeaway

The strongest empirical pattern from the March 26 loop is:

- the winner stayed fixed
- the runner-up changed with seed and search configuration

That is the expected signature of a real recoverable structure rather than overfit noise.
