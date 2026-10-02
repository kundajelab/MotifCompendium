# `composite.py`: the modified Smith-Waterman aligners, in detail

This document describes `MotifCompendium/utils/composite.py`: what problems it solves, the two
scoring models, the two dynamic programs (DPs), and every function. It was written against the
2646-line version of the file on `next_version` (2026-10-02). Line numbers refer to that version.

The file holds two aligners that share their preprocessing and helpers:

| Part | Engine | Used by | Subunits |
|---|---|---|---|
| I | the fixed-length state DP (`calculate_modified_smith_waterman`) | `run_modified_smith_waterman_alignment` (ZNF), `run_modified_smith_waterman_znf`, `run_modified_smith_waterman_composite_fixlen` | one length M, chained in stack order, the full penalty set |
| II | the variable-length chain DP (`_chain_composite`) | `run_modified_smith_waterman_composite` | each trimmed to its own length, ordered by position, one overlap penalty |

A note on names. The code says *finger* (zinc-finger alignment) and *subunit* (composite
motifs) for the same thing: a short motif. Part I says **finger**, Part II says **subunit**.

### Changes since the previous version of this document

The previous version described the 2207-line file. Since then:

| Area | Before | Now |
|---|---|---|
| Composite wrapper | `run_modified_smith_waterman_composite`, fixed M, stack order | renamed `run_modified_smith_waterman_composite_fixlen`, unchanged; a **new** `run_modified_smith_waterman_composite` trims subunits to their own lengths and orders them by position (Part II) |
| Signed similarity (`label_unsigned=False`) | 4-channel cosine of signed motifs | **8 channels** (each base's positive and negative part apart), as MotifCompendium's similarity (Section 3.2) |
| Number of matches | chosen by `match_count_penalty` / `match_count_fraction` | also **limited** by `match_count_max` (Sections 12, 22) |
| Exact DP speed | every diagonal, an argmax per source state, every row filled, one strided argmax per count | **exact shortcuts** for zero overlap/gap and finger-skip penalties (Section 8.6); 7–10× faster on the composite |
| IC scaling | the global IC-scale setting, switched off inside | an **`ic_scale` argument**: the composite wrappers IC-scale the target and the subunits once, at entry (default True); the alignment wrappers don't (default False) (Section 3.1) |
| Trimming argument | `motif_trim` | renamed **`subunit_motif_trim`** |
| Frozen 3 bp recurrence | kept | still kept (the user's decision), penalties ≥ 0 |

---

## Contents

1. [The problems](#1-the-problems)
2. [Map of the file](#2-map-of-the-file)
3. [Preprocessing and similarity](#3-preprocessing-and-similarity)

**Part I: the fixed-length engine**

4. [The alignment model and its score](#4-the-alignment-model-and-its-score)
5. [The DP grid](#5-the-dp-grid)
6. [Moves and candidates](#6-moves-and-candidates)
7. [The DP states](#7-the-dp-states)
8. [The exact DP: filling the grid](#8-the-exact-dp-filling-the-grid)
9. [`general_penalty_root`: rescaling by the number of matches](#9-general_penalty_root-rescaling-by-the-number-of-matches)
10. [Finding the best alignment, tracing it back, and scoring it](#10-finding-the-best-alignment-tracing-it-back-and-scoring-it)
11. [The approximate DP (`global_exact=False`)](#11-the-approximate-dp-global_exactfalse)
12. [Choosing and limiting the number of matches](#12-choosing-and-limiting-the-number-of-matches)
13. [`calculate_modified_smith_waterman`: the driver](#13-calculate_modified_smith_waterman-the-driver)
14. [`_run_modified_smith_waterman`: local, then remaining global](#14-_run_modified_smith_waterman-local-then-remaining-global)
15. [Rebuilding the aligned motif](#15-rebuilding-the-aligned-motif)
16. [The fixed-length wrappers](#16-the-fixed-length-wrappers)
17. [The frozen 3 bp recurrence](#17-the-frozen-3-bp-recurrence)

**Part II: the variable-length composite**

18. [Why a second engine](#18-why-a-second-engine)
19. [Subunits, candidates and match scores](#19-subunits-candidates-and-match-scores)
20. [The alignment model: order by position, energy-weighted overlap](#20-the-alignment-model-order-by-position-energy-weighted-overlap)
21. [`_chain_composite`: the chain DP](#21-_chain_composite-the-chain-dp)
22. [`run_modified_smith_waterman_composite`: the wrapper](#22-run_modified_smith_waterman_composite-the-wrapper)

**Both**

23. [Utilities and plotting](#23-utilities-and-plotting)
24. [Cost](#24-cost)
25. [Observations and caveats](#25-observations-and-caveats)

---

## 1. The problems

The inputs are:

| Input | Shape | Meaning |
|---|---|---|
| target motif | (L, D) | A CWM, usually D = 4 (A, C, G, T) |
| finger / subunit stack | (F, M, D) | F short motifs, padded to a common length M |

The question: what is the best way to explain the target as a **left-to-right chain of short
motifs**? Each placed motif earns a **match score** (how well it fits its window of the
target); overlaps, gaps and skips can cost **penalties**.

| Wrapper | Stack | Order and orientation | Engine |
|---|---|---|---|
| `run_modified_smith_waterman_alignment` (ZNF) | a zinc-finger protein's fingers, as one (F·M, D) matrix | fixed: the fingers as given, in order | Part I |
| `run_modified_smith_waterman_composite_fixlen` | F subunit motifs of length M | an expanded stack allows reordering, reverse-complementing and repeats (Section 16.1) | Part I |
| `run_modified_smith_waterman_composite` | F subunit motifs, padded to M | each trimmed to its own length; any order, orientation and repeats, by position (Part II) | Part II |

The fixed-length engine is Smith-Waterman with one generalization: each finger occupies M target
positions, so a match can come from M different diagonals (Section 6). The variable-length engine
is a chain of weighted intervals with an overlap cost (Section 21).

---

## 2. Map of the file

| Lines | Section | Functions |
|---|---|---|
| 18–192 | Variable-length composite wrapper | `run_modified_smith_waterman_composite` |
| 195–356 | Fixed-length composite wrapper | `run_modified_smith_waterman_composite_fixlen` |
| 359–524 | Variable-length helpers and DP | `_trim_subunits`, `_check_match_count_max`, `_first_running_max`, `_chain_composite` |
| 527–687 | Alignment wrappers | `run_modified_smith_waterman_alignment`, `run_modified_smith_waterman_znf` (deprecated alias) |
| 690–946 | Fixed-length core runner | `_run_modified_smith_waterman`, `_next_unused_indices` |
| 952–1002 | Unrolling, similarity, shuffle | `_unroll_motif`, `compute_aligned_similarity`, `shuffle_motif_position` |
| 1008–1119 | Penalty check; helpers for the frozen recurrence and the action strings | `_previous_state`, `_check_penalty`, `_num_overlap`, `_calculate_overlap_penalties`, `_overlap_actions`, `_diagonal_actions`, `_calculate_finger_skip_penalty`, `_extract_best_alignment` |
| 1122–1744 | Exact DP ("DP states") | `_EMPTY_STATE`, `_state_space`, `_state_value`, `_match_rescale`, `_initial_parts`, `_offer`, `_row_candidates`, `_cell_candidates`, `_fill_modified_smith_waterman`, `_best_cell`, `_traceback_step`, `_traceback`, `_path_state`, `_path_actions`, `_state_matrices`, `_trace_all_states`, `_best_by_count`, `_select_match_count`, `_max_matches`, `_align_all_states` |
| 1747–1984 | Approximate DP ("one state per cell") | `_one_state_row_candidates`, `_one_state_cell_candidates`, `_fill_one_state`, `_one_state_matrices`, `_align_one_state` |
| 1987–2170 | Fixed-length driver | `calculate_modified_smith_waterman` |
| 2173–2427 | Frozen original | `calculate_modified_smith_waterman_znf` |
| 2430–2535 | Reconstruction | `get_alignment_modified_smith_waterman`, `_alignment_tracks` |
| 2540–2646 | Plotting | `visualize_motif`, `visualize_motifs_n` |

Call graph of a fixed-length run:

```
run_modified_smith_waterman_alignment / _znf / _composite_fixlen
└── _run_modified_smith_waterman
    ├── _unroll_motif, compute_aligned_similarity          (match scores; 8 channels if signed)
    ├── calculate_modified_smith_waterman(local)           (the DP)
    │   ├── _align_all_states  (global_exact=True)
    │   │   ├── _state_space
    │   │   ├── _fill_modified_smith_waterman
    │   │   │   ├── _row_candidates   (diagonal + down, one row at a time)
    │   │   │   └── _cell_candidates  (right + start, one cell at a time)
    │   │   ├── _best_cell / _best_by_count
    │   │   └── _trace_all_states → _traceback
    │   ├── or _align_one_state   (global_exact=False)
    │   │   ├── _state_space  (same tables)
    │   │   ├── _fill_one_state → _one_state_row_candidates, _one_state_cell_candidates
    │   │   └── _traceback
    │   ├── _select_match_count  (with a profile)
    │   ├── _path_state    (replays the moves: msw_state, msw_score)
    │   └── _path_actions  (moves → action strings)
    ├── _next_unused_indices
    ├── calculate_modified_smith_waterman(global)          (remaining fingers; ZNF only)
    └── get_alignment_modified_smith_waterman → _alignment_tracks
        + utils_similarity.compute_similarities            (sim_score)
```

Call graph of a variable-length run:

```
run_modified_smith_waterman_composite
├── _trim_subunits                                         (each subunit to its own length)
├── _unroll_motif, compute_aligned_similarity              (per subunit length; 8 channels if signed)
├── _chain_composite                                       (the chain DP)
│   └── _first_running_max
├── _select_match_count, _match_rescale
└── utils_similarity.compute_similarities                  (sim_score)
```

---

## 3. Preprocessing and similarity

### 3.1 Steps (fixed-length: `_run_modified_smith_waterman`, lines 767–814)

| Step | Code | Effect |
|---|---|---|
| Global IC scaling off | `utils_config.set_ic_scale(False)`, restored at the end (line 767) | `compute_similarities` (for `sim_score`) never IC-scales again |
| IC-scale | `ic_scale`: `utils_motif.ic_scale` on target and fingers, once (line 775) | Similarities, match scores and the rebuilt motif all use the IC-scaled motifs |
| Unsigned | `label_unsigned`: `np.abs` on target and fingers (line 780) | Negative CWM contributions count like positive ones |
| Pad | `np.pad(target, ((M-1, M-1), (0, 0)))` (line 785) | A finger may hang off either end, overlapping the target by as little as 1 bp |
| Normalize target | `target / ‖target‖_F` (line 792) | The whole padded target has unit Frobenius norm |
| Normalize fingers | each finger `/ ‖finger‖_F`, all-zero fingers stay zero (line 794) | Each finger has unit norm |
| Unroll | `_unroll_motif(target, M)` → (L−M+1, M, D) (line 797) | Every length-M window of the target |
| Similarities | `compute_aligned_similarity`, on 8 channels when signed (lines 808–814) | Cosine similarity of each finger with each window, forward and reverse-complemented |

`ic_scale` is an argument of every wrapper: True by default for the two composite wrappers,
False for the alignment and ZNF wrappers (their B1H finger PPMs usually come IC-scaled). The
fixed-length composite IC-scales before it expands the stack and passes `ic_scale=False` to
`_run_modified_smith_waterman`. `utils_motif.ic_scale` multiplies each position by its
information content, so it is not idempotent: a motif scaled before and again here is scaled
twice (intended, for example, for references that start as PPMs).
The variable-length wrapper does the same per subunit length (Section 19). From here on in
Part I, **L is the padded length**, L_orig + 2(M−1). The number of windows is L − M + 1, which
the code calls `L_M_1`. Target position t is padded position t + M − 1.

### 3.2 Similarity: 4 or 8 channels

Let $T$ be the padded target after normalization ($\lVert T\rVert_F = 1$), $W_l = T[l:l+M]$ its
window at position l, and $\hat f_i$ finger i after normalization.

$$
\text{sim}^{\text{for}}_{i,l} = \frac{\langle \phi(\hat f_i),\ \phi(W_l)\rangle}{\lVert \hat f_i\rVert\,\lVert W_l\rVert},
\qquad
\text{sim}^{\text{rev}}_{i,l} = \frac{\langle \phi(\hat f_i),\ \phi(\mathrm{rc}(W_l))\rangle}{\lVert \hat f_i\rVert\,\lVert W_l\rVert}
$$

$\mathrm{rc}$ reverses both the positions and the channels (`[:, ::-1, ::-1]`). With channels in
A, C, G, T order, reversing the channels is complementing.

**$\phi$, the channel map.**
- With `label_unsigned=True` (the default), $\phi$ is the identity on $|x|$: 4 channels.
- With `label_unsigned=False`, $\phi$ = `utils_motif.motif_4_to_8`. It splits each base into its
  positive part $\max(x, 0)$ and its negative part $\max(-x, 0)$, in 8 channels laid out as
  A+, A−, C+, C−, G−, G+, T−, T+.

Two facts make the 8-channel version a drop-in replacement:

1. **Norms are unchanged.** Every entry is either positive or negative, so it lands in exactly
   one channel: $\lVert\phi(x)\rVert = \lVert x\rVert$.
2. **Reverse-complementing still works as `[::-1, ::-1]`.** The layout pairs each channel with
   its complement at the mirrored index (A+ ↔ T+, A− ↔ T−, C+ ↔ G+, C− ↔ G−). So
   $\phi(\mathrm{rc}(x)) = \mathrm{rc}(\phi(x))$, which section i of the tests checks.

What changes is the inner product. On 4 signed channels a positive finger and a negative window
give a negative product, and opposite signs cancel. On 8 channels the positive and negative
parts are compared separately and never cancel, so a **negative region only matches a negative
finger**. This is the comparison `assign_label_from_other_compendium(label_unsigned=False)` makes.

For unsigned motifs there is no negative part, so 8 channels give the same similarities as 4.
The code then keeps 4 channels, so that unsigned results stay bit for bit as before.

### 3.3 Window importance and match scores (Part I)

**Window importance** (`scale_factor`, called `target_motif_norm_unroll_importance`, line 799):

$$
s_l = \lVert W_l \rVert_F^2
$$

The windows of a tiling of the target (non-overlapping, covering it) have importances that sum to
$\lVert T\rVert^2 = 1$.

**Match score** of finger i at window l (in `calculate_modified_smith_waterman`, line 2135):

$$
m_{i,l} = s_l \cdot \text{sim}_{i,l}
$$

A perfect tiling therefore scores about 1. The penalties and `match_count_penalty` are on this
scale: `match_count_penalty = 0.02` keeps a match only if it raises the score by more than 0.02.

**Matchable** cells: $\text{sim}_{i,l} >$ `sim_threshold` (which must be < 1). Only matchable
cells can host a match.

### 3.4 Which version of the target is used where

| Use | Version | Does the normalization matter? |
|---|---|---|
| Windows for the similarities | normalized | No: `compute_aligned_similarity` normalizes each window again |
| Window importance $s_l$ | normalized | **Yes**: a full-motif norm of 1 makes a tiling's importances sum to 1 |
| `matrix_l` passed to the driver and to `get_alignment_modified_smith_waterman` | normalized | No: only its shape is read |
| Remaining global alignment | normalized, sliced from `remain_l`, **not** normalized again | Its importances stay on the scale of the whole target |
| `sim_score` | **padded, not normalized** | No: `compute_similarities` normalizes both motifs internally |

One consequence of normalizing the whole motif: a target with a long, weak flank gives its strong
core windows importances summing to less than 1. A perfect alignment of the core then scores
less than 1, so fixed penalties weigh relatively more for that target.

---

# Part I: the fixed-length engine

## 4. The alignment model and its score

### 4.1 What an alignment is

An alignment is a sequence of **moves** through a grid of (finger, window) cells (Section 5).
Its matches are pairs (finger i, window l) with both i and l strictly increasing: fingers are
used **in stack order**, and a finger may be skipped.

For two consecutive matches at windows $l_j$ and $l$, the distance is $d = l - l_j$:

| d | Relation | Penalty charged at the second match |
|---|---|---|
| 1 … M−1 | overlap of o = M − d bp | $o_1 + (o-1)\,o_2$ |
| M | abutting | 0 |
| ≥ M+1 | gap of G = d − M bp | $g_1 + (G-1)\,g_2$ |

| Parameter | Symbol | Meaning |
|---|---|---|
| `overlap_penalty_f1` | $o_1$ | first overlapping bp |
| `overlap_penalty_f2` | $o_2$ | each further overlapping bp |
| `skip_penalty_l1` | $g_1$ | first bp of a gap |
| `skip_penalty_l2` | $g_2$ | each further bp of a gap |
| `skip_penalty_f1` | $k_1$ | the first finger of a run of skipped fingers |
| `skip_penalty_f2` | $k_2$ | each further finger of the run |

Every penalty is **affine** (one price for the first unit, another for each further unit), is
given as a **number ≥ 0**, and is **subtracted**. `_check_penalty` (line 1020) raises
`ValueError` for a negative number (the old sign convention) and for the removed string value
`'l2_norm'`.

| | 1 unit | 2 | 3 | u units |
|---|---|---|---|---|
| overlap of u bp | $o_1$ | $o_1+o_2$ | $o_1+2o_2$ | $o_1 + (u-1)o_2$ |
| gap of u bp | $g_1$ | $g_1+g_2$ | $g_1+2g_2$ | $g_1 + (u-1)g_2$ |
| run of u skipped fingers | $k_1$ | $k_1+k_2$ | $k_1+2k_2$ | $k_1 + (u-1)k_2$ |

Nothing before the first match is charged. Whether anything after the last match is charged
depends on the alignment type (Section 25, item 6).

### 4.2 The score

$$
\boxed{\;H \;=\; r(N)\cdot H_\text{match} \;-\; P_\text{overlap} - P_\text{skip,f} - P_\text{skip,l}\;}
\qquad
r(N) = \begin{cases} N^{-1/p} & p = \texttt{general\_penalty\_root} > 0\\ 1 & p = 0\end{cases}
$$

- $H_\text{match} = \sum m_{i,l}$ over the N matches.
- The three penalty sums are each ≥ 0.

`msw_state` reports exactly these: `H_match`, `H_penalty_overlap_f`, `H_penalty_skip_f`,
`H_penalty_skip_l` (each ≥ 0), and `F_total` = N. When the number of matches is chosen
(Section 12), `msw_score` is $H - \texttt{match\_count\_penalty}\cdot N$.

### 4.3 What the rescaling means

`_match_rescale(n, p)` (line 1265) returns $n^{-1/p}$, with a fast path of `1/sqrt(n)` for p = 2,
and 1 for p = 0.

- **p = 2** divides by √N, the norm of N unit vectors that don't overlap. It re-L2-normalizes
  the sum of the placed fingers.
- **p = 1** divides by N, so the score becomes the average match.
- **p = 0** turns the rescaling off.
- N equal matches of score s give $N s \cdot N^{-1/p} = s\,N^{1-1/p}$. With p = 2 that grows like √N.

**When does one more match help?** An alignment with n matches has match sum H. Adding a match of
score s (ignoring penalties) raises the score if and only if

$$
\frac{H+s}{(n+1)^{1/p}} > \frac{H}{n^{1/p}}
\iff
s > H\left[\left(1+\tfrac1n\right)^{1/p} - 1\right] \approx \frac{H}{p\,n} = \frac{\bar m}{p}
$$

where $\bar m$ is the average match so far. A new finger must beat about 1/p of the average
match.

---

## 5. The DP grid

`_fill_modified_smith_waterman` (line 1407) allocates

```
parts: (F + 1, num_cols, *state_shape, num_parts)      num_cols = L + M   (L padded)
```

| Grid index | Meaning |
|---|---|
| row `i = f + 1`, f = 0 … F−1 | finger f |
| column `j = l + M`, l = 0 … L−M | window l (visited columns: M … L) |
| row 0, and columns 0 … M−1 | "before the matrix": never visited, hold the **initial state** |
| columns L+1 … L+M−1 | unused (M−1 columns) |

Every cell starts as the initial state, at (L_skip = −1, F_skip = 0, F_total = initial), with
parts from `_initial_parts` (line 1278). The fill then **overwrites** every visited cell with
the best candidates it computes; only the unvisited cells keep the initial state.

- For **local** alignment, the initial state is `_EMPTY_STATE` (line 1124): all sums 0, F_total = 0.
- For the **remaining global** alignment, it is the local alignment's `msw_state` (Section 14).

`_initial_parts` converts a state dict to the parts the DP stores (Section 7.5):
- 1 part: $H_\text{match} - (P_\text{overlap} + P_\text{skip,f} + P_\text{skip,l})$
- 2 parts: $[H_\text{match},\ -(P_\text{overlap} + P_\text{skip,f} + P_\text{skip,l})]$

The first M columns exist so that diagonal k (which reads column j − k) never goes out of range
for any k ≤ M. `H_final` in the returned matrices has shape (F+1, L+M) with padded L, i.e.
(F+1, L_orig + 3M − 2).

---

## 6. Moves and candidates

A cell (i, j) = (finger f, window l) can be reached by M + 3 kinds of move. They are listed in
**tie-break order**: when two candidates have the same value, the earlier one wins, because
`_offer` only replaces a candidate that is strictly better.

| Move id | Name | From | Charge (added to the penalty part) | Effect on the state |
|---|---|---|---|---|
| 0 … M−1 | Diagonal k = 1 … M: **match** | (i−1, j−k) | $+m_{f,l}$ to the match part; −overlap and −gap for $d = L_\text{skip} + k$ | L_skip → 0, F_skip → 0, F_total → F_total + 1 |
| M | Down: **skip finger** f | (i−1, j) | $-k_1$ if F_skip = 0, else $-k_2$ | F_skip → 1 (i.e. "1 or more") |
| M+1 | Right: **skip window** l | (i, j−1) | $-g_2$ only if L_skip is already M+1 | L_skip → L_skip + 1 (−1 stays −1; M+1 stays M+1) |
| M+2 | Start: **new local alignment** at this match | nothing | parts = (m) or (m, 0) | L_skip 0, F_skip 0, F_total 1 |

Diagonal moves and Start require the cell to be matchable. Start also requires a local alignment.

**Why diagonal k.** A match at window l, coming from (i−1, j−k), means the previous row's
alignment was last at window l − k; the k − 1 windows in between are passed over. If the
previous row's state says the last match was $L_\text{skip}$ windows before that, the distance
between the two matches is $d = L_\text{skip} + k$.

**When diagonals k ≥ 2 can't win.** Without L_skip states (no overlap or gap penalty), a Right
move is free and keeps the state. Row i−1 then never decreases along j, so the source of
diagonal 1, (i−1, j−1), is at least as good as the source of any diagonal k, (i−1, j−k). The
match adds the same $m_{f,l}$ and no distance-dependent charge, so diagonal 1 is never beaten;
it is also offered first and wins ties. The code therefore tries only diagonal 1 in that case
(Section 8.6). With L_skip states, diagonal k from (i−1, j−k) gives the same value as k−1 Right
moves followed by diagonal 1, and the capped gap accounting splits the cost identically; an
earlier check on 80 random cases with the ZNF defaults found no score or action list changed when
the loop was restricted to k = 1. The code still tries every k there.

---

## 7. The DP states

### 7.1 Why states are needed

Classic Smith-Waterman keeps one number per cell. That fails here because the cost of the *next*
move depends on how the cell was reached:

- The overlap or gap penalty of the next match depends on how many windows ago the last match was.
- The next finger skip costs $k_1$ or $k_2$ depending on whether a skip run is already open.
- With rescaling, the value of H_match depends on how many matches there are.

An alignment that looks worse at a cell can be the better one to continue. So the exact engine
keeps **the best alignment for each state** in each cell. `_state_space` (line 1128) defines the
states and the tables both engines use:

| Axis | Values | Index | Tracked when |
|---|---|---|---|
| L_skip | −1 (no match yet), 0, 1, …, M, M+1 (meaning "M+1 or more") | L_skip + 1, so 0 … M+2 | any of $o_1, o_2, g_1, g_2$ is nonzero (`track_l`, line 1185) |
| F_skip | 0, 1 (meaning "1 or more") | 0, 1 | $k_1 \ne k_2$ (`track_f`, line 1187) |
| F_total | 0 … num_n − 1 | same | set by the caller (Sections 9, 11, 12) |

Shape: `(M+3 or 1, 2 or 1, num_n)`. An axis that no penalty depends on has a single value, so zero
penalties cost no extra state. `states` gives the flat index of each state.

### 7.2 Why L_skip has values up to M+1

The gap penalty applies to the gap at the **next** match, $G = L_\text{skip} + k - M$, where k is
chosen later. The state must remember enough to price **every** k = 1 … M.

Two L_skip values can share one state only if their penalty-versus-k curves differ by a constant,
because that constant can be charged at the Right move. The charge for (k, s) is

$$
\text{pen}(k, s) =
\begin{cases}
o_1 + (M - s - k - 1)\,o_2 & s + k < M \quad\text{(overlap of } M-s-k \text{ bp)}\\
0 & s + k = M\\
g_1 + (s + k - M - 1)\,g_2 & s + k > M
\end{cases}
$$

**s ≤ M−1.** Some k gives s + k = M (abut, 0), and k + 1 then gives a 1 bp gap ($g_1$). Each s has
this kink at a different k, so these values must stay separate (unless $g_1 = g_2$).

**s ≥ M.** Every k gives a gap, and $\text{pen}(k, s) = g_1 + (s - M)\,g_2 + (k-1)\,g_2$, linear in
s with slope $g_2$ whatever k is. The code caps L_skip at **M+1** and charges $g_2$ for each
further window as it is passed:

| Where | Charge |
|---|---|
| Right move out of L_skip = M+1 (line 1394) | $g_2$ per window |
| Diagonal k from the capped state (`skip_l[k, M+2]`) | $g_1 + k\,g_2$ |
| Total, for a true L_skip of s ≥ M+1 | $(s-M-1)\,g_2 + g_1 + k\,g_2$ ✓ |

Under the affine rule the linear regime already starts at s = M, so the cap could be at M (one
state fewer); the M+1 state is a leftover of the old gap rule (Section 25, item 3). L_skip = −1
means no previous match: its charges are 0, and Right keeps it at −1.

### 7.3 The tables for M = 3

Printed from `_state_space(3, o1=0.02, o2=0.1, k1=0.1, k2=0.1, g1=0.02, g2=0.1, …)`, the ZNF
wrapper's defaults. The tables hold what a move **adds** to the score (minus the penalty). Rows
are k and columns are L_skip.

**Overlap** (`overlap_f`):

| k \ L_skip | −1 | 0 | 1 | 2 | 3 | 4 (= M+1) |
|---|---|---|---|---|---|---|
| 1 | 0 | −0.12 (d=1, 2 bp) | −0.02 (d=2, 1 bp) | 0 | 0 | 0 |
| 2 | 0 | −0.02 (d=2, 1 bp) | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 |

**Gap** (`skip_l`):

| k \ L_skip | −1 | 0 | 1 | 2 | 3 | 4 (= M+1) |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 (d=3, abut) | −0.02 (1 bp) | −0.12 (2 bp) |
| 2 | 0 | 0 | 0 (abut) | −0.02 (1 bp) | −0.12 (2 bp) | −0.22 (3 bp) |
| 3 | 0 | 0 (abut) | −0.02 (1 bp) | −0.12 (2 bp) | −0.22 (3 bp) | −0.32 (4 bp) |

Plus −0.1 ($-g_2$) per Right move out of L_skip = 4.

### 7.4 The moves, as tables by state

`_state_space` also precomputes each move's result for every state, as lists indexed by flat
state, which the approximate engine uses directly (Section 11):

| Table | Value for state (d, f_skip, n) |
|---|---|
| `d_of`, `count` | its L_skip index d and its F_total n |
| `diag_to` | state after a match: (d0, 0, n+1); `None` if n+1 is past the last F_total; (d0, 0, 0) if F_total isn't tracked |
| `down_to`, `down_pen` | (d, nF−1, n); $-k_2$ if f_skip = 1, else $-k_1$ |
| `right_state`, `right_pen` | untracked or d = 0: stays, no charge; d < M+2: (d+1, f_skip, n), no charge; d = M+2: stays, $-g_2$ |
| `start_state` | (d0, 0, 1), or (d0, 0, 0) if F_total isn't tracked |

The exact engine uses the equivalent index arrays `right_from = [0, 1, …, M+1]` →
`right_to = [0, 2, …, M+2]`, plus `skip_f1`, `skip_f2`, `skip_l2` (the charges, negated). When
$k_1 = k_2$ the F_skip axis collapses (`nF = 1`) and every skip costs $k_1$.

### 7.5 What a state stores: 1 or 2 parts

| Case | Parts | Value (`_state_value`, line 1254) |
|---|---|---|
| no rescaling (root 0), or rescaling with all penalties 0 | `("H",)`: $H_\text{match}$ − penalties | $H \cdot w$ |
| rescaling **and** some penalty (`split=True`) | `("H_match", "H_penalty")`, with $H_\text{penalty}$ = −(sum of penalties) ≤ 0 | $H_\text{match}\cdot w + H_\text{penalty}$ |

Only H_match is rescaled, so the two must be kept apart exactly when there is both a rescaling
and a penalty. `pen` is the index of the part that penalties go to: 0 with one part, 1 when
split. The three separate penalty sums are **not** stored; for the alignment found, they are
recovered by replaying its moves (Section 10.4).

---

## 8. The exact DP: filling the grid

### 8.1 Loop structure

```python
for f in range(F):                                   # rows
    if free_down and not matchable[f].any():         # Section 8.6
        parts[f + 1] = parts[f]; continue
    row = _row_candidates(space, parts, f+1, M, M+L_M_1, ...)   # diagonal + down, whole row, vectorized
    for l in range(L_M_1):                           # columns, left to right
        best = _cell_candidates(space, parts, f+1, l+M, ..., [x[l] for x in row])   # + right + start
        parts[f+1, l+M] = best[1]
```

Diagonal and Down moves read only row i−1, which is already complete, so they are computed for
the whole row at once. Right reads (i, j−1) from the same row, so it is done cell by cell.

### 8.2 `_offer`: keep the better candidate, state by state

```python
value, parts, move, source = best
candidate_value = _state_value(candidate, weight)
better = candidate_value > value[target]          # strictly: earlier candidates win ties
value[target]  = where(better, candidate_value, value[target])
parts[target]  = where(better, candidate, parts[target])
move[target]   = where(better, candidate_move, move[target])
source[target] = where(better, candidate_source, source[target])
```

`target` selects the destination states. All comparisons use `space.weight`, the match weight of
the pass (Section 9).

### 8.3 `_row_candidates` (line 1299): diagonals and Down

For each k = 1 … M (only if some cell in the row is matchable; only k = 1 without L_skip states):

1. `candidate = parts[i-1, j0-k : j1-k]`, the source cells for the whole row.
2. Add $m_{f,l}$ to part 0, and `overlap_f[k][L_skip]` + `skip_l[k][L_skip]` to part `pen`.
3. Set part 0 to −∞ for cells that aren't matchable.
4. **Pick the best source state.** A match sends every source (L_skip, F_skip) to the same target
   (L_skip = 0, F_skip = 0), so per cell and per F_total take the argmax over the flattened
   sources (first maximum on ties). With a single source state, this is skipped.
5. Offer it to the target (L_skip index `d0`, F_skip 0, F_total + 1), with move k − 1. The F_total
   shift is `n_from = 0 … nN−2` → `n_to = 1 … nN−1` (line 1319): the top F_total has no
   successor, which is how a pass caps the number of matches (Sections 9.5, 12).

Then Down: `below = parts[i-1, j0:j1]`, with $-k_1$ (from F_skip 0) or $-k_2$ (from F_skip 1)
added to part `pen`, into F_skip 1, with move M.

### 8.4 `_cell_candidates` (line 1353): Right and Start

- **Right** (move M+1): `left = parts[i, j-1]`, shifted `right_from → right_to`; the capped index
  M+2 goes to itself with $-g_2$ on part `pen`. Without L_skip states it is a straight copy.
- **Start** (move M+2, local only, matchable only): the fresh parts (m) or (m, 0) at
  (L_skip 0, F_skip 0, F_total 1). The traceback stops at a Start.

**Why the traceback can recompute moves.** Both the fill and the traceback call
`_row_candidates` + `_cell_candidates` on the same stored `parts`, so the traceback gets back
exactly the moves and sources the fill chose. No matrix of moves is stored (Section 10.2).

### 8.5 The empty alignment

The first M columns are never visited, so they hold the initial state. In a local alignment that
is the empty alignment, value 0, at L_skip −1. A Right move from column M−1 carries it into every
row for free, so every row has a value-0 option: the `max(0, ·)` of classic Smith-Waterman. A
diagonal from the empty state gives m with no charge, the same value as Start, and wins the tie.

### 8.6 Exact shortcuts

Three shortcuts skip work whose outcome is already determined. Each applies only when the
penalties make it exact, and each leaves every output bit for bit the same.

| Shortcut | Applies when | Why it is exact | Where |
|---|---|---|---|
| Only diagonal 1 | no L_skip states (overlap and gap penalties 0) | each row never decreases along j, so diagonal 1 is never beaten and wins ties (Section 6) | `range(1, M + 1 if nD > 1 else 2)`, line 1322 |
| No argmax over source states | one source state per F_total (`nD * nF == 1`) | there is nothing to choose | line 1329 |
| Copy rows that can't match | no F_skip state and $k_1 = 0$ (`free_down`) | with no match, a cell takes the cell above (Down, first) or its left neighbour, which the row above already took; so the row equals the row above | lines 1430–1434 |

`_best_by_count` (line 1616) also scans the score matrix DP row by DP row instead of with one
`argmax(axis=0)` over the flattened matrix, which is strided and was the largest single cost on
long stacks. Updating only on a strictly greater value keeps the first maximum, as `np.argmax`.

Together these made the fixed-length composite 7–10× faster on real ENCODE motifs against the
1487-reference clustered stack (CTCF_pos_high_0: 80 s → 11 s), with identical outputs on every
motif checked and the full test suite passing.

---

## 9. `general_penalty_root`: rescaling by the number of matches

### 9.1 The difficulty

$$
H = N^{-1/p}\,H_\text{match} - P
$$

N is the **final** number of matches, which a partial alignment in the middle of the grid doesn't
know yet. The score is not a sum of per-move terms, so ordinary DP doesn't apply directly.

### 9.2 The rule that makes DP work

At each cell and state, the DP keeps one partial alignment and discards the others. That is safe
only if:

> whenever partial A beats partial B at a cell and state, A still beats B after **any**
> continuation they both can take.

Write a continuation as: c more matches, match sum h, penalty sum q (≥ 0). A partial with f
matches, match sum H and penalties P finishes at $\text{final} = w(f+c)\,(H+h) - P - q$, with
$w(n) = n^{-1/p}$.

### 9.3 The three regimes (`_align_all_states`, line 1659)

`general = root != 0` and `split = general and any(penalties)`. Each pass is a tuple
`(num_n, match_weight, final_weight, final_f_total)`:

| Regime | Condition | Passes | F_total axis | Parts | Weight during the fill | Weight at the end | States kept at the end |
|---|---|---|---|---|---|---|---|
| A | root = 0 | 1 | 1 (or all counts, with a profile) | 1 | 1 | 1 | all |
| B | root ≠ 0, all penalties 0 | 1 | `num_f_total` | 1 | 1 | `rescale[F_total]` | all |
| C | root ≠ 0, some penalty ≠ 0 | one per n = max(F_total₀, 1) … num_f_total−1 | n + 1 | 2 (split) | w(n) | w(n) | F_total = n only |

$$
\texttt{num\_f\_total} = F_{\text{total},0} + \min(F,\ L-M+1,\ \texttt{match\_count\_max}) + 1
\qquad (\texttt{\_max\_matches}, \text{line 1651})
$$

Every finger matches at most once (one per row) and every match uses at least one new window, so
an alignment has at most min(F, L−M+1) matches; `match_count_max` caps it further (Section 12).

### 9.4 Regime B: no penalties, rescale at the end

With every penalty 0, the single part is H_match. Two partials at the same cell with the **same**
count f end with the same count f + c under the same continuation, so

$$
\text{A} - \text{B} = w(f+c)\,(H_A - H_B),
$$

whose sign is that of $H_A - H_B$ whatever c turns out to be. Keeping the larger raw H_match is
safe, and w doesn't need to be known during the fill.

| Continuation | P1 (f = 2, H = 1.3) | P2 (f = 2, H = 1.1) |
|---|---|---|
| stop (c = 0) | 1.3/√2 = 0.919 | 1.1/√2 = 0.778 |
| +1 match of 1.0 | 2.3/√3 = 1.328 | 2.1/√3 = 1.212 |
| +3 matches, total 2.7 | 4.0/√5 = 1.789 | 3.8/√5 = 1.699 |

Partials with **different** counts can't be compared, because the same continuation gives them
different weights:

| Continuation | P1 (f = 2, H = 1.3) | P3 (f = 1, H = 0.9) | winner |
|---|---|---|---|
| stop | 1.3/√2 = 0.919 | 0.9/√1 = 0.900 | P1 |
| +1 match of 1.0 | 2.3/√3 = 1.328 | 1.9/√2 = 1.344 | **P3** |

So F_total is a state axis, and each cell stores the best raw H_match separately for every count.
A match moves a partial to slot f+1 and a skip keeps it in slot f, so the two never compete; the
choice is made at the end, where `score = _state_value(parts, rescale)` with
`rescale = [1, w(1), w(2), …]` broadcast over the F_total axis. The maximum over every cell and
slot is exact.

### 9.5 Regime C: penalties, one pass per final count

With penalties, two partials with the same F_total differ by
$w(N)\,(H_A - H_B) - (P_A - P_B)$, and the sign depends on N. Example, f = 2: A has H = 2.0,
P = 0.2; B has H = 1.6, P = 0:

| Final N | w(N) = 1/√N | A − B = 0.4·w − 0.2 | Keep |
|---|---|---|---|
| 2 | 0.707 | +0.083 | A |
| 3 | 0.577 | +0.031 | A |
| 4 | 0.500 | 0 | tie |
| 6 | 0.408 | −0.037 | B |

**The fix: guess N.** For each n, one full fill assumes the final alignment has exactly n matches:

1. Weight every match by w(n) during the fill (`space.weight = w(n)`, split parts). The difference
   is now a fixed number, so the DP rule holds again.
2. At most n matches: the F_total axis has n + 1 values, and the diagonal shift has no successor
   for F_total = n.
3. Exactly n at the end: visited cells with F_total ≠ n are set to −∞ (line 1712).

The best over all passes (line 1730, strict `>`, passes in increasing n) is the global optimum,
and ties go to fewer fingers.

| | Regime B | Regime C |
|---|---|---|
| Fills | 1 | ≈ K = min(F, L−M+1, `match_count_max`) |
| Parts per state | 1 | 2 |
| What slot f holds | best partial for **every** final N | best partial **assuming** final N = n |
| When w is applied | at the end: H·w(F_total) | on every match, during the fill |

---

## 10. Finding the best alignment, tracing it back, and scoring it

### 10.1 `_best_cell` (line 1443)

| Alignment | Where it may end | Rule |
|---|---|---|
| local | any cell and state, including unvisited cells (the empty alignment, 0) | first maximum of `score.argmax()` in flat order |
| global | the last row or the last column | first maximum over those cells, in that order |

If no alignment scores above 0, the argmax is cell (0, 0), the traceback loop never runs, and the
result is the empty alignment.

### 10.2 `_traceback` (line 1487), shared by both engines

```python
while (i, j) is a visited cell:
    move, source = moves_at(i, j)      # engine-specific: recompute the cell's candidates
    option = move[row];  row = source[row]
    step, i, j, started = _traceback_step(option, ...)
    if started: break
```

`row` is a flat state index for the exact engine and a row index for the approximate engine.
`_traceback_step` (line 1467) turns a move into a step and a source cell:

| Move | Step | Source cell |
|---|---|---|
| k − 1 < M | `("match", f, l, k)` | (i−1, j−k) |
| M | `("F-skip", f, l, 0)` | (i−1, j) |
| M+1 | `("L-skip", f, l, 0)` | (i, j−1) |
| M+2 | `("match", f, l, 0)`, and stop | – |

### 10.3 `_path_actions` (line 1552): steps → action strings

The rest of the code works with action strings `F{finger}@{action}@L{window}`:

| Action | Emitted for |
|---|---|
| `F{f}@match@L{l}` | a match |
| `F{f}@F-overlap@L{…}` | each overlapped position before a match (`_overlap_actions`) |
| `F{f}@L-skip@L{l-i}` | each window a diagonal k steps over, and each Right move |
| `F{f}@F-skip@L{l}` | a Down move |

`_diagonal_actions` builds [overlaps…, match, skipped windows…], dropping windows before the start
of the matrix (`guard_negative=True`; the frozen recurrence passes False).

### 10.4 `_path_state` (line 1511): the score, by replaying the moves

The DP keeps only one or two numbers per state, so the separate sums of the alignment found (its
`msw_state`) are rebuilt by charging each move exactly as the DP did:

| Step | Update | Tracked |
|---|---|---|
| start (`k == 0`) | reset to `_EMPTY_STATE`, then treat as a match with k = 1 from L_skip −1 | |
| match | H_match += m; overlap −= overlap_list[k][d]; skip_l −= skip_list[k][d]; F_total += 1 | d ← d0, f_skip ← 0 |
| F-skip | skip_f += $k_2$ if a run is open (and $k_1 \ne k_2$), else $k_1$ | f_skip += 1 |
| L-skip at d = M+2 | skip_l += $g_2$ | |
| L-skip at 0 < d < M+2 | – | d += 1 |

It returns the state dict and
$\text{score} = H_\text{match}\cdot r(F_\text{total}) - P_\text{overlap} - P_\text{skip,f} - P_\text{skip,l}$.
The driver uses `_state_space(M, *penalties, 1, 1.0)` ("rules", line 2162) for the tables.

### 10.5 `_state_matrices` (line 1573; only with `return_matrices`)

`H_final[i, j]` is the best score over all states at the cell; `state_final` holds the stored
parts with only the state axes that have more than one value, plus `axes` giving the value of
each index.

---

## 11. The approximate DP (`global_exact=False`)

`_align_one_state` (line 1929) keeps **one alignment per cell**, or, with a profile, one per cell
and F_total ("rows"), on the same `_state_space` tables as the exact engine. Each row stores
`parts` (1 or 2 numbers, or `None`), `states` (its flat state) and `score`.

| Move | Source row | New parts | New state |
|---|---|---|---|
| Diagonal k | `row − 1` with a profile, else `row` | + m, + overlap_list[k][d] + skip_list[k][d] (to part `pen`) | `diag_to[s]`; skipped if `None` |
| Down | `row` | + `down_pen[s]` | `down_to[s]` |
| Right | `row` | + `right_pen[s]` if not `None` | `right_state[s]` |
| Start | – (row 1 with a profile, else 0) | (m) or (m, 0) | `start_state` |

A row's value is $\text{parts}[0]\cdot \text{rescale}[\text{count}[s]]\ (+\ \text{parts}[1])$, using
the **current** F_total of the state.

**Why it can miss the best alignment.** Keeping one alignment per cell discards alternatives that
would continue better (a cheaper L_skip for the next overlap, no open skip run), and ranking by
the current F_total uses $w(f)$ rather than the final $w(N)$ (in Section 9.4 it keeps P1 and
drops P3). It is exact only when every penalty is 0 **and** the root is 0.

**Profile mode delegates.** With a profile, if no penalty depends on L_skip or F_skip and the
parts aren't split, the exact engine is both exact and cheaper, so `_align_one_state` calls
`_align_all_states` (line 1951). Since `match_count_max` (default 3 in the composite wrappers)
turns the profile on, the fixed-length composite's defaults (all penalties 0, root 2) now always
take this exact path, although `global_exact=False` (Section 25, item 1).

---

## 12. Choosing and limiting the number of matches

Profile mode is on when `return_match_profile`, `match_count_penalty != 0`,
`match_count_fraction < 1`, or `match_count_max` is not None (line 2126). It is only allowed for
local alignment.

$$
\texttt{msw\_profile}[N] = \max\{\,H(\text{alignment}) : \text{exactly } N \text{ matches}\,\},
\qquad \texttt{msw\_profile}[0] = 0
$$

(−∞ where no alignment has exactly N matches.)

**Limit** (`match_count_max`, validated by `_check_match_count_max`, line 385: None or an
integer ≥ 1). The F_total axis stops at `match_count_max` (`_max_matches`, Section 9.3), so the
DP never builds an alignment with more matches. `msw_profile` then stops at it, and the alignment
found is the best (exactly, with `global_exact`) with at most that many matches. A limit of
min(F, L−M+1) or more changes nothing (section h of the tests). It also shrinks the DP: at 3, the
composite ran 2.5–3× faster on real motifs.

**Selection** (`_select_match_count`, line 1640), with `match_count_penalty` c ≥ 0:

$$
\text{pen}(N) = \texttt{msw\_profile}[N] - c\cdot N,
\qquad
N^* = \min\{\,N : \text{pen}(N) \ge \texttt{match\_count\_fraction}\cdot\max_{N'} \text{pen}(N')\,\}
$$

With fraction 1 this is the N that maximizes pen, taking the fewest matches on ties. N = 0 is
always a candidate. **`msw_score` includes the count penalty** (line 2165):
$\texttt{msw\_score} = H(\text{chosen}) - c\cdot N^*$; `msw_profile` and `msw_state` are reported
without it. With `global_exact`, this is exactly the best alignment under a per-match penalty c,
found from one profile without a DP pass per final count. Across orientations, per N, the later
orientation wins ties (`>=`, line 2150).

---

## 13. `calculate_modified_smith_waterman`: the driver

Lines 1987–2170.

1. **Validate.** Each of the six penalties and `match_count_penalty` go through `_check_penalty`
   (≥ 0); `match_count_max` through `_check_match_count_max`. Also checked: `sim_threshold < 1`,
   M > 1, `general_penalty_root ≥ 0`, the range of `match_count_fraction`, and no profile for
   global alignment.
2. `initial_state` = the given dict or `_EMPTY_STATE` (H_match, the three penalty sums, F_total).
3. `align = _align_all_states if global_exact else _align_one_state`.
4. For each orientation: `match_scores = scale_factor * sim`, `matchable = sim > sim_threshold`,
   then `align(..., profile=profile, max_count=match_count_max)` with `num_cols = L + M`.
5. Pick the result: without a profile, the best score over orientations (the later one winning
   ties); with a profile, Section 12.
6. **Replay**: `msw_state, msw_score = _path_state(...)`, minus c·N with a profile.
7. `msw_actions = _path_actions(steps, …)`.

Returns `(msw_score, msw_orient, msw_actions, msw_state, H_final, state_final, msw_profile)`.
"rev" compares every finger with reverse-complemented windows, which is the same as
reverse-complementing every finger; the chain still runs left to right in stack order.

---

## 14. `_run_modified_smith_waterman`: local, then remaining global

Line 690.

1. Preprocess (Section 3).
2. **Local** alignment over all fingers and the whole padded target (line 817).
3. **Remaining global** alignment (line 848). It runs only if `align_remaining`, the local
   alignment is non-empty, and fingers and windows are left over:
   - `_next_unused_indices` (line 919): `remain_f` = the last action's finger + 1, `remain_l` = the
     end of the last match's footprint.
   - A **global** alignment of fingers `remain_f:` against windows `remain_l:`, in the local
     alignment's orientation, from `initial_state = msw_state`, without a profile or limit.
   - Its actions are appended. Its **score is discarded**, but **`msw_state` is replaced** by the
     combined state.
4. **Rebuild** (Section 15), with `rescale_factor = _match_rescale(F_total, root)` (line 892).
5. `sim_score` = MotifCompendium similarity between the sum of the placed fingers and the padded
   target (line 906).
6. Restore the IC-scale setting.

Returns `(msw_score, sim_score, aligned_motif_fs, msw_orient, msw_align_f, msw_align_l,
msw_scale_f, msw_actions, H_final, state_final[, msw_profile])`.

| Wrapper | `align_remaining` | `place_as_matched` |
|---|---|---|
| ZNF / alignment | True: every finger is accounted for | False: fingers placed as given |
| composite (fixed length) | False: leftover stack entries are unused copies | True: placed in the orientation matched |

---

## 15. Rebuilding the aligned motif

### 15.1 `get_alignment_modified_smith_waterman` (line 2430)

For each `match` action (finger i, window l):

$$
\text{aligned}[i,\ l:l+M] = \hat f_i \cdot s_l \cdot \rho,
\qquad \texttt{msw\_scale\_f}[i] = s_l\cdot\rho,
\qquad \rho = F_\text{total}^{-1/p}\ (1 \text{ for } p = 0)
$$

`aligned_motif_fs` has shape (F, L, D) in padded coordinates; the sum over fingers is the
reconstructed target. (With fingers of one length, scaling by the window energy only rescales
the reconstruction; Part II scales by the projection instead, Section 22.)

### 15.2 `_alignment_tracks` (line 2472)

Two equal-length tracks, one entry per column: `msw_align_f` (finger column f·M + k, or `'-'`)
and `msw_align_l` (target position, or `'-'`). A matched finger's M columns sit over the positions
it covers; shared positions stay under the **earlier** finger; an unmatched finger faces `'-'`.
Every finger column and every target position appears exactly once, in order.

---

## 16. The fixed-length wrappers

### 16.1 `run_modified_smith_waterman_composite_fixlen` (line 195)

**Defaults:** all penalties 0, `general_penalty_root=2`, `global_exact=False`,
`match_count_penalty=0`, `match_count_fraction=1`, `match_count_max=3`, `label_unsigned=True`,
`allow_reverse_order=True`, `allow_reverse_comp=True`, `max_homo_subunits=2`, `ic_scale=True`
(the target and subunits are IC-scaled once, before the stack is expanded, line 295).

**The expanded stack.** Matches must follow stack order, so the stack decides which arrangements
are possible. It is made of blocks of all F subunits (line 307):

```python
orders  = [0..F-1] (+ [F-1..0] if allow_reverse_order)
orients = [0]      (+ [1]      if allow_reverse_comp)
blocks  = [(order, rc) for order in orders for rc in orients] * max_homo_subunits
```

For F = 2 subunits A and B, with the defaults (′ = reverse complement):

| Block | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| (order, rc) | (fwd, 0) | (fwd, 1) | (rev, 0) | (rev, 1) | (fwd, 0) | (fwd, 1) | (rev, 0) | (rev, 1) |
| entries | A B | A′ B′ | B A | B′ A′ | A B | A′ B′ | B A | B′ A′ |

That is 8F entries. With `allow_reverse_comp`, both "for" and "rev" passes run; "rev" flips every
entry once more, so one copy allows both forward-then-reverse and reverse-then-forward.

**Mapping back.** For each match action (stack entry e, window l):

| Output | Value |
|---|---|
| `aligned_subunit_idx` | `stack_subunit_idx[e]` |
| `aligned_subunit_alignment_rc` | `stack_rc[e] XOR (msw_orient == "rev")` |
| `aligned_subunit_alignment_h` | l, a padded column (the target starts at column M−1) |
| `aligned_subunit_scale` | `msw_scale_f[e]`, including $N^{-1/p}$ |
| `aligned_subunit_stack` | (N, 2M + L_orig − 2, D): each match's placed subunit at its column |

### 16.2 `run_modified_smith_waterman_alignment` (line 527)

**Defaults:** $o_1 = 0.02$, $o_2 = 0.1$, $k_1 = k_2 = 0.1$, $g_1 = 0.02$, $g_2 = 0.1$,
`general_penalty_root=0`, `global_exact=True`, `match_count_max=None`, `finger_length=3`,
`ic_scale=False`. It
reshapes the (F·M, D) reference into (F, M, D) and calls the core with `align_remaining=True` and
fingers placed as given. `match_count_max` limits the local alignment only; the remaining
alignment can still add fingers.

### 16.3 `run_modified_smith_waterman_znf` (line 642)

A deprecated alias for 16.2 with the same arguments. It does not use the frozen recurrence.

---

## 17. The frozen 3 bp recurrence

`calculate_modified_smith_waterman_znf` (line 2173) is the original algorithm, kept only to
reproduce old results; the tests use it as a gate against the original code. It requires M = 3.
It takes penalties as numbers ≥ 0 (through `_check_penalty`) and negates them once, so the
original arithmetic is unchanged, and it keeps the **old** gap rule: $g_1$ for a 1 bp gap, else
$(G-1)\,g_2$.

| Aspect | Frozen recurrence | Current DP |
|---|---|---|
| Per cell | one score H and a full action list A | 1–2 parts per state; moves recomputed |
| Penalties | found by scanning the source cell's action list for the last match | carried in the state |
| Gap rule | $g_1$, or $(G-1)g_2$ | $g_1 + (G-1)g_2$ |
| Known bugs (docstring) | on diagonals 2 and 3, an abutting finger is charged a gap penalty for its distance to the finger before the previous one; diagonal 1 never charges a gap penalty | fixed |

Its helpers (`_previous_state`, `_calculate_overlap_penalties`, `_calculate_finger_skip_penalty`,
`_extract_best_alignment`) are used only by it.

---

# Part II: the variable-length composite

## 18. Why a second engine

Composite subunits are reference motifs, stored padded to a common width (42 bp for the
clustered references), with cores of very different lengths (median 13 bp after trimming, range
3–38). The fixed-length engine treats every subunit as M bp wide. Three problems follow:

1. **Credit for energy outside the core.** A match is credited with the energy of its whole
   M-bp window. With M ≥ the target's length, most windows hold nearly the whole target, so a
   reference that loosely resembles the whole composite outscores a precise subunit at one end.
2. **Two subunits on the same site.** Two references whose cores sit at different offsets in
   their frames reach the same target site from different windows, and both get full credit.
   The overlap penalties measure overlap between frames, and almost every pair of 42-bp frames
   overlaps, real composite or not.
3. **Stack order.** Arrangements are possible only if the expanded stack (8F entries) encodes
   them, and every entry is a DP row.

Each of these comes from the fixed length M, which sets the windows, the diagonals 1…M, the
overlap rule M − d, the L_skip states and the output frame. The composite case needs none of the
per-step penalty states. With subunits trimmed to their own lengths and matches ordered by
position, the problem becomes a chain of weighted intervals, which a much simpler exact DP solves
(Section 21).

---

## 19. Subunits, candidates and match scores

### 19.1 Trimming (`_trim_subunits`, line 359)

`subunit_motif_trim` has the meaning of `logo_trimming` in `assign_label_from_other_compendium`:

| `subunit_motif_trim` | Each subunit becomes |
|---|---|
| `False` | the whole M positions |
| `True` (default) | `utils_motif.trim_motif(subunit, 1/M)`: flanks trimmed at the standard threshold 1/M |
| a number in [0, 1] | `utils_motif.trim_motif(subunit, subunit_motif_trim)`; 0 trims only all-zero positions |

`trim_motif` keeps the positions between the first and last position whose share of the
subunit's total absolute weight exceeds the threshold. A subunit trimmed away entirely raises a
`ValueError` naming it; other values of `subunit_motif_trim` raise a `ValueError` too. Trimming
runs on the subunits after IC scaling (`ic_scale`) and `np.abs` (when unsigned), so flanks are
trimmed by their IC-scaled weight.

### 19.2 Candidates

A **candidate** c is a subunit, as given or reverse complemented (`allow_reverse_comp`), with its
own length $M_c$, L2-normalized: $\hat c$. Candidates are listed subunit by subunit
(`cand_subunit`, `cand_rc`), so there are F or 2F.

### 19.3 Windows, similarity and match scores (lines 138–158)

The target is padded by M − 1 (the untrimmed width) on either side and L2-normalized as a whole
($\lVert T\rVert = 1$), as in Part I. A candidate of length $M_c$ at padded start l covers the
window $W^{(c)}_l = T[l:l+M_c]$, so it may overhang either end. For each distinct length m, the
windows of length m are unrolled once and compared with all candidates of that length in one
matmul (`compute_aligned_similarity`, 8 channels when signed, Section 3.2):

$$
\text{sim}_{c,l} = \frac{\langle \phi(\hat c),\ \phi(W^{(c)}_l)\rangle}{\lVert W^{(c)}_l\rVert},
\qquad
e_{c,l} = \lVert W^{(c)}_l\rVert^2,
\qquad
m_{c,l} = e_{c,l}\cdot \text{sim}_{c,l}.
$$

A start is **matchable** if its window lies within the padded target and
$\text{sim}_{c,l} >$ `sim_threshold`. The match score is the fixed-length one, except that the
window is the subunit's own extent: a candidate is credited only for the target energy it covers.
Non-overlapping candidates cover disjoint energy, so their match scores sum to at most 1.

---

## 20. The alignment model: order by position, energy-weighted overlap

### 20.1 Ordering by position

An alignment is up to `match_count_max` candidates $(c_1, l_1), \dots, (c_N, l_N)$ with
**strictly increasing starts** $l_1 < \dots < l_N$. Nothing else constrains the sequence:

| | Fixed length (stack order) | Variable length (position order) |
|---|---|---|
| A then B, or B then A | only via the order blocks | always |
| A then A′ (opposite orientations) | via the orientation blocks and the "rev" pass | always |
| the same subunit twice | at most `max_homo_subunits` times | any number, up to `match_count_max` |
| arguments | `allow_reverse_order`, `max_homo_subunits` | removed; `allow_reverse_comp` stays |

### 20.2 What an overlap double counts

Let consecutive candidates a then b cover windows $[l_a, t_a)$ and $[l_b, t_b)$ with
$l_a < l_b$, and write $E(X)$ for the target energy of a set of positions X. Their match scores are

$$
m_a + m_b = \text{sim}_a\,E([l_a, t_a)) + \text{sim}_b\,E([l_b, t_b)).
$$

If they overlap, on $O = [l_b, \min(t_a, t_b))$, the energy $E(O)$ is in both windows, while the
two together only cover $E([l_a, t_a)) + E([l_b, t_b)) - E(O)$. With perfect similarities, the
alignment is credited with $E(O)$ too much. That is how two subunits on one site scored above 1 in
the fixed-length engine. Charging

$$
\text{overlap cost}(a, b) = \lambda\cdot E(O), \qquad \lambda = \texttt{overlap\_penalty}
$$

with **λ = 1 (the default) takes the doubly counted energy back once**: for perfect matches the
score is then exactly the energy the two subunits cover together. With similarities below 1 the
double count is $E(O)$ times about the smaller similarity, so λ = 1 charges slightly more than
that; λ > 1 discourages overlaps further, and λ = 0 allows them for free.

A constant per-bp penalty can't do this: a motif's energy is concentrated in its cores, so the
energy per core bp is well above the average 1/L. A per-bp penalty small enough not to forbid
legitimate overlaps on weak flanks lets two subunits share a core almost for free. This was
measured: at 0.05 per bp, with root 0, second subunits sat 1–4 bp from the first on several real
composites.

The cost is the energy of the positions of b that the **previous** candidate covers, measured
with the cumulative position energy $W_x = \sum_{y < x} \lVert T_y \rVert^2$:

$$
\text{overlap cost} = \lambda\,\bigl(W_{\min(t_a, t_b)} - W_{l_b}\bigr) \quad\text{if } t_a > l_b,\ \text{else } 0.
$$

Only consecutive candidates are compared. If a long subunit contains the next two, the third's
overlap with the first isn't charged; charging it exactly would need the union of all earlier
extents as state.

### 20.3 The score

$$
\boxed{\;H = r(N)\sum_{i} m_{c_i, l_i} \;-\; \lambda \sum_{i \ge 2} E(O_{i-1,i})\;}
\qquad r(N) = N^{-1/p}\ (1 \text{ for } p = 0)
$$

with the rescaling of Section 4.3. The number of matches is chosen and limited as in Section 12,
and `msw_score` is $H - \texttt{match\_count\_penalty}\cdot N$.

**Root and count penalty on this scale.** Because match scores of non-overlapping subunits now
sum to at most 1, root 2 rarely pays for a second subunit: two subunits with match scores
$e_1 \ge e_2$ beat the first alone only if $(e_1 + e_2)/\sqrt2 > e_1$, i.e. $e_2 > 0.414\,e_1$,
and a `match_count_penalty` c raises that to $e_2 > 0.414\,e_1 + 1.414\,c$. With root 0, each
subunit adds its own match score, and c reads as "a subunit must explain more than c of the
target's energy".

---

## 21. `_chain_composite`: the chain DP

Line 404.

### 21.1 Recurrence

Process the matchable (candidate, start) pairs by increasing start l. Let
`best[e, n]` be the best score so far of an alignment with n matches whose **last** candidate
ends at e, over alignments whose last start is before l. A candidate c at l, ending at
$t = l + M_c$, with match score $m$, extends the best alignment with n − 1 matches whose last
candidate ends at e, at a cost depending on where e falls:

| Previous end e | Overlap | Value |
|---|---|---|
| $e \le l$ | none | $\max_{e \le l} \texttt{best}[e, n-1]$ |
| $l < e \le t$ | $[l, e)$ | $\max_{l < e \le t}\bigl(\texttt{best}[e, n-1] - \lambda W_e\bigr) + \lambda W_l$ |
| $e > t$ | all of $[l, t)$ | $\max_{e > t} \texttt{best}[e, n-1] - \lambda (W_t - W_l)$ |

$$
V(c, l, n) = m + \max(\text{the three cases}), \qquad V(c, l, 1) = m.
$$

Each case is a running maximum over e, so all candidates at one start are scored with a few
vectorized passes over the end positions: a prefix maximum, a running maximum of
$\texttt{best} - \lambda W$ from l+1, and a suffix maximum. `_first_running_max` (line 393)
returns each running maximum with the index of its first occurrence, so the traceback is
deterministic. Ties go to less overlap (the cases in table order), then to the earliest end, then
to the earliest pair.

After all candidates at l are scored, they are added to `best[t, n]`, so they are visible only to
later starts; this enforces strictly increasing starts. Each value stores the pair that preceded
it at that moment (`previous`), because `best[e, n−1]` can later be replaced by an alignment that
starts at or after l.

**Why it is exact.** The cost of extending an alignment depends only on its last candidate's end
(and on n), so for each (e, n) only the best alignment ending at e matters; the three cases
partition the possible ends. The number of matches is bounded by the number of distinct starts
with a matchable candidate (starts strictly increase), and by `match_count_max`.

### 21.2 The rescaling

With $r(N) \ne 1$ and λ > 0, the score $r(N) H_\text{match} - P$ is not a sum of per-candidate
terms. As in regime C (Section 9.5), one pass per final N maximizes
$H_\text{match} - P / r(N)$, i.e. runs the DP with λ/r(N), and reports $r(N)$ times the best value
with exactly N matches. Without a penalty (λ = 0) or without the rescaling (root 0), one pass
suffices: the best $H_\text{match}$ by n is rescaled at the end, as in regime B (line 499). With
`match_count_max` = 3 this is at most 3 passes.

### 21.3 Output

`msw_profile[n]` (exactly n matches; 0 for n = 0, −∞ if impossible), and `trace(n)`: the
(candidate, start) pairs of the best alignment with n matches, left to right.

### 21.4 Checks

- Brute force over every chain of ≤ N candidates on 150 random small cases (mixed lengths, a
  per-bp or a random energy weight, λ from 0 to 0.9, root 0, 2 and 1.5, limits None, 1, 2, 3):
  every profile equal to within 1e-12, and every traced chain re-scoring to it (section j of the
  tests).
- With `subunit_motif_trim=False` and λ = 0, the allowed chains are exactly those of the fixed-length
  wrapper with enough copies, all penalties 0 and `global_exact=True` (any order and orientation,
  strictly increasing starts, any overlap). The scores agreed to 1e-12 in 48 of 48 random cases.

---

## 22. `run_modified_smith_waterman_composite`: the wrapper

Line 18.

**Defaults:** `subunit_motif_trim=True`, `sim_threshold=0`, `overlap_penalty=1`,
`general_penalty_root=2`, `match_count_penalty=0`, `match_count_fraction=1`, `match_count_max=3`,
`label_unsigned=True`, `allow_reverse_comp=True`, `ic_scale=True`. There is no `global_exact`:
the DP is always exact.

1. Validate: `overlap_penalty` and `match_count_penalty` through `_check_penalty`,
   `match_count_max` through `_check_match_count_max`, the root, the fraction and the threshold.
2. Global IC scaling off (restored at the end, line 111); with `ic_scale`, the target and the
   subunits are IC-scaled once (line 116); `np.abs` on both when unsigned (line 121).
3. Trim (Section 19.1) and build the candidates (19.2).
4. Pad and normalize the target; similarities and match scores per length (19.3).
5. `position_energy` = $\lVert T_x\rVert^2$ per padded position (sums to 1, line 161), the
   overlap weight.
6. `_chain_composite`, then `_select_match_count`; `msw_score` = `msw_profile[N]` − c·N.
7. Place each matched candidate (trimmed, oriented, normalized) at its start, scaled by
   $\langle \hat c, W\rangle\, r(N) = \lVert W\rVert\,\text{sim}\; r(N)$ (line 173).
8. `sim_score` = MotifCompendium similarity between the sum of the placed subunits and the padded
   target.

**Why the projection, not the window energy.** $\langle \hat c, W\rangle$ is the least-squares
weight of $\hat c$ on its window. For a target made exactly of the subunits, the placed subunits
sum to the target (up to the common factor r(N)), whatever their lengths, so `sim_score` is 1.
Scaling by the window energy $e$, as the fixed-length engine does, weights each subunit by its
energy share; with different lengths that distorts the reconstruction (planted composites then
gave `sim_score` 0.993–0.996 instead of 1).

**Outputs** (the fixed-length composite's tuple, in the same coordinates):

| Output | Value |
|---|---|
| `msw_score` | best score, minus `match_count_penalty` · N |
| `sim_score` | MotifCompendium similarity of the collapsed alignment with the target |
| `aligned_subunit_stack` | (N, 2M + L − 2, D): each matched subunit, trimmed, oriented, scaled, at its start; the target spans columns M−1 … L+M−2 |
| `aligned_subunit_idx` | subunit index into the input stack |
| `aligned_subunit_alignment_rc` | 0 forward, 1 reverse complement |
| `aligned_subunit_alignment_h` | start column (padded; target position = h − (M−1)), of the trimmed subunit |
| `aligned_subunit_scale` | projection × $N^{-1/p}$ |
| `msw_profile` | best score by N, before the count penalty (only with `return_match_profile`) |

So `h - (ref_len - 1)` still gives the start on the target, with `ref_len` the stack's width.

### 22.1 On real composites

Ten ENCODE composites against the 1487 IC-scaled clustered references (`sim_threshold` 0.5,
`match_count_max` 2), measured before `ic_scale` was added (so the target was not IC-scaled),
compared with `assign_label_from_other_compendium` (match, then subtract,
`max_submotifs=2`):

| Setting | Same subunit set as the labeler | Two subunits | Mean `sim_score` | Time per motif |
|---|---|---|---|---|
| fixed length, notebook settings (root 2, c 0.25) | 2/10 | 7/10 | 0.875 | 5.8 s |
| variable length, root 2, c 0, λ 1 | 5/10 | 1/10 | 0.853 | 0.07 s |
| variable length, root 0, c 0.2, λ 1 | 5/10 | 2/10 | 0.879 | 0.07 s |
| variable length, root 0, c 0.1, λ 1 | 5/10 | 5/10 | **0.929** | 0.07 s |
| variable length, root 0, c 0.1, λ 2 | 4/10 | 4/10 | 0.909 | 0.07 s |

The variable-length engine is about 80× faster, places no reference twice on one site, and
reports starts at the subunits' cores. With root 0, c 0.1 and λ 1, two of the three TEAD::AP1
composites get an AP-1-family reference, BATF_0 (with ARID_0), and NFI::FOX gets NFI_1 + a FOX
reference; with root 2, a second subunit is rarely worth it (Section 20.3). The labeler is not ground truth: for GATA::GATA_pos_high_1
the second GATA2 site (30 bp from the first) that the fixed-length engine found looks like a real
dimer the labeler misses.

---

# Both

## 23. Utilities and plotting

| Function | What it does |
|---|---|
| `_unroll_motif(motif, M, xp)` | (L, K) → (L−M+1, M, K), gathering the windows with an index array; works with numpy or cupy |
| `compute_aligned_similarity(x, y)` | (N, M) cosine similarities of flattened, normalized (N, L, K) and (M, L, K) tensors, for any K (4, or 8 for signed motifs); GPU matmul, returned as numpy; all-zero motifs give 0 |
| `_first_running_max(values)` | running maximum and the index of its first occurrence |
| `shuffle_motif_position(motif)` | randomly permutes the positions of an (L, 4) motif (a null model) |
| `visualize_motif(motif, path)` | a logomaker logo of one (L, 4) motif, saved to a file |
| `visualize_motifs_n(motifs, path)` | N logos stacked vertically, with shared y-limits and dashed lines at each motif's first/last nonzero position |

The module imports `cupy` at the top level, so it needs a GPU environment (for example
`motifcompendium-gpu`) even for code paths that never touch the GPU.

---

## 24. Cost

Notation: K = min(F, L−M+1, `match_count_max`), $n_D$ = M+3 or 1, $n_F$ = 2 or 1.

| Engine, mode | Fills | States per cell | Work per fill |
|---|---|---|---|
| Fixed, exact, regime A | 1 | $n_D n_F$ (× K+1 with a profile) | $O(F\,L\,M\,n_D n_F)$ ($O(F\,L)$ diagonals without L_skip states) |
| Fixed, exact, regime B | 1 | K+1 | $O(F\,L\,K)$; rows that can't match are copied |
| Fixed, exact, regime C | K | $n_D n_F (n+1)$ in pass n, 2 parts | $O(F\,L\,M\,n_D n_F\,K^2)$ in total |
| Fixed, approximate | 1 | 1 (K+1 rows with a profile) | $O(F\,L\,M)$, in pure Python |
| Variable length | 1 (or K with root and λ) | – | $O(S \cdot K \cdot S)$ vectorized + $O(P \cdot K)$ insertions, S padded positions, P matchable pairs |

The fixed-length engine computes everything for each orientation (2 passes with
`allow_reverse_comp`), and each fixed-length composite row is one of 8F stack entries. The
variable-length engine has one row per candidate (2F), no orientation passes, and no per-cell
Python loop. On the 1487-reference stack, a typical ENCODE motif takes about 4 s with the
fixed-length engine (1–3.5 s with `match_count_max=3`) and 0.07 s with the variable-length one.

---

## 25. Observations and caveats

1. **`global_exact=False` is rarely approximate now.** The fixed-length composite's defaults
   (penalties 0, root 2, `match_count_max=3`) turn the profile on, and `_align_one_state` then
   delegates to the exact engine (Section 11). It is a heuristic only with penalties that depend
   on L_skip or F_skip, or with rescaling and penalties together.
2. **Diagonals k ≥ 2 never win without L_skip states**, and the code skips them (Section 8.6).
   With L_skip states they are still tried; an earlier check found them redundant there too.
3. **The L_skip = M+1 state is no longer needed.** Under the affine gap rule, L_skip = M and M+1
   differ by exactly $g_2$ for every k (Section 7.2). Capping at M would save one state.
4. **The global continuation starts at L_skip = −1**, so no overlap or gap penalty is charged
   between the last local match and the first remaining match. `remain_l` starts after the last
   local match's footprint, so an overlap can't happen, but the gap is free.
5. **`msw_score` and the rebuilt motif use different N** in the ZNF wrapper: `msw_score` is the
   local alignment's, while the placed fingers are scaled with the combined F_total.
6. **Global alignment must end in the last row or column.** Trailing skips are charged on the way
   there, so "nothing after the last match is charged" holds for local alignment only.
7. **Variable length: only consecutive overlaps are charged** (Section 20.2). With
   `match_count_max` ≤ 3 a subunit nested inside an earlier, longer one is the only uncharged
   case.
8. **Variable length: settings need re-tuning.** Match scores of non-overlapping subunits sum to
   at most 1, so settings tuned on the fixed-length engine (whose credits exceeded 1) select
   fewer subunits (Section 20.3).
9. **The frozen recurrence keeps the old gap rule** (Section 17). Regression comparisons against
   it are only meaningful with zero gap penalties.
10. **Ties** go consistently to: earlier moves (diagonal before down before right before start),
    smaller k, earlier source states, the earliest cell (local), fewer fingers (regime C), the
    **later** orientation ("rev") in Part I; less overlap, the earliest end and the earliest pair
    in Part II.
