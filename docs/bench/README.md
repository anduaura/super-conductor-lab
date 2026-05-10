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

## Ambient-pressure landscape (`ambient.csv`) — the actual goal

`mode="ambient"` evaluates Tc at operating pressure ≈ 1 atm regardless of
the candidate's synthesis pressure. High-pressure-only superconductors
(LaH₁₀ regime) score near zero. The four peaks are placed at
(h_frac, en_diff, avg_val) combinations only:

| peak | h_frac | en_diff | avg_val | height |
| --- | ---: | ---: | ---: | ---: |
| anomalous valence | 0.45 | 1.5 | 0.5 | 250 K |
| high-H + common valence | 0.85 | 1.0 | 2.0 | 220 K |
| cuprate-like (S-rich) | 0.30 | 2.0 | 1.0 | 270 K |
| **closest to RTSC** | **0.50** | **1.6** | **1.5** | **305 K** |

This is the landscape that actually points the optimizer at the project's
north star — ambient-pressure Tc ≥ 293 K. Same 8 × 10 grid as above:

| strategy | n | median | p25 | p75 | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ucb** | 10 | **237.8 K** | 132.3 | 249.1 | **311.3 K** |
| ei | 10 | 233.9 K | 167.5 | 248.1 | 299.6 K |
| ucb+manifold | 10 | 191.4 K | 151.8 | 241.9 | **317.9 K** |
| ucb+inverse | 10 | 189.9 K | 133.1 | 248.3 | 266.4 K |
| ucb+falsify | 10 | 172.9 K | 148.0 | 227.0 | 279.7 K |
| all | 10 | 155.7 K | 136.2 | 198.0 | **317.9 K** |
| thompson | 10 | 146.9 K | 115.5 | 180.0 | 268.1 K |
| **random** | 10 | **112.6 K** | 83.7 | 149.8 | 194.7 K |

Three observations:

1. **The RTSC threshold (293 K) was crossed.** Three strategies (`ucb`,
   `ucb+manifold`, `all`) hit best-Tc above 293 K on at least one seed —
   the synthetic landscape says these compositions qualify as
   ambient-pressure RTSC.
2. **No strategy reliably gets there.** Median Tc tops out at 237.8 K
   (`ucb`); the high-Tc hits are best-of-10 outliers, not consistent
   discovery. The closed loop *can* find Peak D when seeded right; it
   doesn't always.
3. **Random search is firmly bottom (median 112.6 K).** The closed loop
   delivers a 2.1× median improvement over random — the largest gap of
   any landscape we've measured.

This is the cleanest expression to date of the project's actual purpose:
the optimizer is *capable* of finding ambient-pressure RTSC candidates in
a landscape that has them, but is not yet *reliable* at it. Closing that
reliability gap is the main motivation for the queued M9–M12 (real
chemistry, learned surrogates, calibrated quantum proxy, literature-
grounded agent).

## Reliability sweep (`reliability.csv`) — quantifying RTSC discovery

The 8×10 grids above answer "median Tc per strategy"; this one answers
**"how often does the loop actually find an ambient-pressure RTSC
candidate?"** — defined as `best Tc ≥ 293 K`. 50 seeds × 10 strategies
× 30 rounds × ambient mode.

| strategy | median | best | **P(≥293 K)** |
| --- | ---: | ---: | ---: |
| **ucb+anneal** (new) | **226.8 K** | **333.4 K** | **12 %** |
| ei | 197.8 K | 330.6 K | 12 % |
| ucb | 215.7 K | 311.3 K | 6 % |
| ucb+manifold | 209.3 K | 317.9 K | 6 % |
| all | 155.1 K | 317.9 K | 4 % |
| thompson | 146.2 K | 327.4 K | 4 % |
| ucb+inverse | 205.3 K | 312.7 K | 2 % |
| ucb+anneal+manifold | 166.0 K | 299.4 K | 2 % |
| ucb+falsify | 167.3 K | 283.7 K | 0 % |
| **random** | **119.9 K** | 239.1 K | **0 %** |

### What we learn

1. **Annealed κ doubles the success rate.** Going from constant κ=2.0 to
   κ linearly decaying from 4.0 (early exploration) to 0.5 (late exploit)
   raises success rate from 6 % → 12 % AND raises best-ever Tc from
   317.9 K → 333.4 K — the highest of any run across all benches.
2. **EI matches `ucb+anneal` on success rate but loses on median.**
   EI has heavier tails (12 % success, but only 197.8 K median); annealed
   UCB has tighter exploitation late in the loop.
3. **Manifold curvature hurts when combined with annealing.** Already-
   exploring early κ + curvature bonus = too much exploration; the loop
   never settles. `ucb+anneal+manifold` is 2 %, vs 12 % for plain
   `ucb+anneal`. Strategy-stacking isn't free.
4. **Random search hits 0 %** even with 50 seeds. The closed loop is
   doing real work; the gap between best Bayesian and random is now
   cleanly quantified.
5. **No strategy clears 293 K reliably.** 12 % is the ceiling we found
   with this architecture and a 30-round budget. The next levers
   (longer horizon, real DFT-trained surrogate, real autonomous lab
   data) are documented as post-M12 open threads.

Reproduce:

```bash
scl bench \
  --strategies "random,ucb,ei,thompson,ucb+manifold,ucb+falsify,ucb+inverse,ucb+anneal,ucb+anneal+manifold,all" \
  --seeds "$(python -c 'print(",".join(str(i) for i in range(1, 51)))')" \
  --rounds 30 \
  --world-mode ambient \
  --out docs/bench/reliability.csv
```

## Notes

- Each row is best-Tc-found from a single seeded run; per-seed variance is
  high, so trust medians + IQR over individual values.
- `all+nnqs` is available as a strategy but is ~10× slower per round
  (the RBM solver runs every six rounds); excluded from the default sweep.
- Total wall time for a full 8 × 10 sweep is ~30 seconds (single) or
  ~50 seconds (multi).
