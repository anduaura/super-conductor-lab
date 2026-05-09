# Benchmark sweeps

Results from `scl bench --seeds 1..10 --rounds 30`, eight strategies on two
ground-truth landscapes. Reproduce with:

```bash
scl bench --world-mode single --out docs/bench/single.csv
scl bench --world-mode multi  --out docs/bench/multi.csv
```

## Single-peak landscape (`single.csv`)

The original landscape — smooth, unimodal, max ≈ 320 K. UCB-style methods
should dominate.

| strategy | n | median | p25 | p75 | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ucb+inverse** | 10 | **291.1 K** | 276.2 | 301.6 | 311.5 |
| ei | 10 | 284.1 K | 273.2 | 291.3 | 310.1 |
| ucb+falsify | 10 | 282.2 K | 262.7 | 288.9 | 305.0 |
| ucb | 10 | 276.9 K | 267.5 | 296.9 | 312.5 |
| thompson | 10 | 273.8 K | 241.0 | 294.6 | 300.3 |
| ucb+manifold | 10 | 263.4 K | 253.6 | 268.3 | 296.8 |
| all (kitchen sink) | 10 | 247.1 K | 237.3 | 256.1 | 276.7 |
| random | 10 | 239.6 K | 220.3 | 274.4 | 282.5 |

On this landscape the data wants to be exploited, not explored: focused
methods (UCB + inverse design, plain EI) win, and the kitchen-sink "all"
configuration actually underperforms plain UCB — adding falsification and
manifold-curvature exploration costs experiment budget the smooth landscape
doesn't reward.

## Multi-modal landscape (`multi.csv`)

Sum of four Gaussian peaks at distinct (h_frac, pressure, en_diff, avg_val)
combinations:

| peak | h_frac | pressure | en_diff | avg_val | height |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (easy) | 0.80 | 100 GPa | 0.8 | 1.5 | 220 K |
| B (good ternary) | 0.85 | 200 GPa | 1.2 | 1.8 | 270 K |
| C (high-pressure binary) | 0.90 | 400 GPa | 0.6 | 2.0 | 260 K |
| **D (best, anomalous)** | **0.85** | **280 GPa** | **1.4** | **0.5** | **320 K** |

Peak D is in a narrow attractor at an unusual valence — the kind of
counterintuitive composition the closed loop is supposed to find.

| strategy | n | median | p25 | p75 | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ucb+manifold** | 10 | **205.0 K** | 197.3 | 212.3 | 218.9 |
| ucb | 10 | 201.7 K | 180.5 | 215.5 | 226.0 |
| ucb+falsify | 10 | 201.7 K | 124.9 | 207.9 | 212.7 |
| ucb+inverse | 10 | 199.5 K | 192.4 | 206.8 | 214.2 |
| all | 10 | 198.6 K | 94.5 | 209.6 | 228.0 |
| ei | 10 | 198.2 K | 87.2 | 206.4 | 226.0 |
| thompson | 10 | 189.7 K | 145.5 | 199.0 | 205.7 |
| random | 10 | 152.0 K | 115.6 | 168.8 | 207.8 |

Three things to read off this table:

1. **Random is bottom across the board** — 50 K behind the best Bayesian
   method. The closed loop is doing real work even on a hard landscape.
2. **UCB + manifold curvature wins on the hard landscape.** The
   curvature-of-belief bonus pays for itself when peaks are separated and
   the surrogate has multiple plausible directions to explore.
3. **No strategy reliably finds Peak D (320 K) in 30 rounds.** Best run
   across all 80 trials: 228 K (still in basin B). This is by design — the
   landscape is calibrated so that 30-round budgets aren't enough, and a
   real S-AGI would need to either run longer, allocate adaptively, or
   incorporate prior structure (e.g. the LLM hypothesizer agent). It's the
   open problem this prototype motivates rather than solves.

## Notes

- Each row is best-Tc-found from a single seeded run; per-seed variance is
  high, so trust medians + IQR over individual values.
- `all+nnqs` is available as a strategy but is ~10× slower per round
  (the RBM solver runs every six rounds); excluded from the default sweep.
- Total wall time for a full 8 × 10 sweep is ~30 seconds (single) or
  ~50 seconds (multi).
