# Implementation Roadmap

Ordered, testable milestones. Each has an explicit **Done when** — a check that can be run, not a
feeling. Hand these to the implementation model one milestone at a time; do not let it run ahead.

The target ruleset is **Blot Star plus the project owner's corrections**
([`01-rules.md`](01-rules.md)). All but two `[OPEN]` items are resolved, and neither remaining one
(OPEN-2c auction termination, OPEN-10 rounding) blocks implementation — both sit behind config
flags. Prereq for everything: `presets/blotstar.yaml` is committed and `rules_hash` is stable.

---

## M0 — Scaffold (0.5 day)

- `pyproject.toml`, package `bazarblot`, Python ≥3.11, `ruff` + `mypy --strict` on `core/`,
  `pytest`, `hypothesis`.
- `presets/blotstar.yaml` + `RuleConfig` dataclass that loads it and exposes `rules_hash`
  (stable hash of the resolved config).
- CI: lint + type + test.

**Done when:** `pytest` passes on an empty suite, `mypy --strict bazarblot/core` is clean, and
`RuleConfig.load("presets/blotstar.yaml").rules_hash` is stable across processes.

---

## M1 — Rules engine (3–5 days) — *the milestone that matters*

Implement `core/`: cards, declarations, auction, play, scoring, deal, match. Pure Python, no
numpy, no torch. Everything deterministic from a seed.

Sub-steps, in order:
1. `cards.py` — ids, orderings, both point tables. Test the §3.3 subtotals (62/30/38/62, 152/152/248).
2. `declarations.py` — sequence detection over *natural* rank order, carrés, Blot-Reblot,
   maximum-value decomposition, cross-team comparison (§6.4) with **elder-hand** tie-breaking
   measured from the trick-1 leader, and Blot-Reblot handled as a flat unconditional +20 that
   sits entirely outside the comparison. Model the **announce/show protocol** (§6.5) in the state
   even though M1 auto-resolves it: separate `announced` from `shown` per seat from the start, so
   turning on `declarations_are_actions` later is a policy change, not a state-model rewrite.
3. `auction.py` — bids as `(level, type, capot_flag)` triples on the `8..80` ladder with **purely
   numeric seniority** (neither `NT` nor capot outranks at equal level), non-binding passes, the
   sticky-capot rule (once anyone bids capot every higher bid must be capot), termination
   (3 passes after a bid → `three_passes_after_bid`; 4 passes → abort), contra/recontra, and a
   hard step cap since non-binding passes let the auction circle indefinitely.
4. `play.py` — legal-move generation for all three contract types incl. ruff/over-ruff
   obligations; trick resolution.
5. `scoring.py` — the Blot Star payout model (rules §7.2): `M × bid + collected + bonuses` on a
   made contract, `M × bid + 16 + bonuses` to the defenders on a failed one, with the bid
   multiplier as a **lookup table** (trump 1/2/4, no-trump 2/3/5 — no single formula reproduces
   it). Combinations count toward fulfilment. Capot **sets** the card portion to 252 rather than
   adding 90, and a capot *bid* additionally requires the shutout. Plus `round10`, applied
   **independently** to each side's raw total — see §7.4's correction: there is no complement
   trick and no invariant that the two scaled scores sum to a constant.
6. `deal.py` / `match.py` — state machines, dealer rotation, target, tiebreaks.

**Tests (this is most of the work — budget for it):**
- Table-driven fixtures for every rules §3.2 card value and every rules §6 combination case,
  including the full §6.4 precedence ladder: carré-beats-sequence-by-category, carré-vs-carré by
  value with a 100/100 elder-hand tie, and the three-way sequence chain
  `A-K-Q` non-trump > `J-10-9` trump > `J-10-9` non-trump.
  Include the Blot-Star-specific carré values explicitly — trump `9 = 140` and `A = 110`, no-trump
  `A = 190` and `9 = 0` — since a developer working from classical Belote will "fix" these to
  150/100 without noticing.
- All three rules §7.5 fixtures: A (ordinary made/failed/failed-under-`NT`), B (a bid of 26
  carried by combinations), C (the capot trap — point target met, shutout missed, contract lost).
- A capot deal that *also* has combinations, to catch a `+= 90` implementation where a `= 252`
  was required. The plain case cannot distinguish them.
- The **maximum-bid deal**: one partner holds the jack and nine carrés, the other the ten and ace
  carrés, and the attacking team takes capot. Construct it by hand, assert the raw total is 802
  and the scaled score is 80, and assert a bid of 80 is both legal and made. This pins the top of
  the ladder and catches scale-then-sum errors (§7.4 requires summing raw and scaling once).
- Hypothesis property tests:
  - card conservation and the §2.3 invariants after every action, on random legal playouts
  - `cards_A + cards_D == 162` **raw** (unscaled) on 10^6 random deals with no capot — this is
    the invariant that actually always holds; do NOT assert it on the *scaled* (`round10`) values,
    which do not sum to a constant (§7.4)
  - the bid payout is never applied to the 16 or to collected points, only to the bid term
  - legal-move set is never empty in `PLAY`
  - a player holding the led suit never has a legal move outside it
  - replay determinism: `(seed, actions)` → identical final state, 10^5 deals
- Golden-file test: 1000 seeded deals played by a fixed pseudo-random policy, final scores hashed.
  Any rules change that alters this hash must be deliberate.

**Done when:** 10^6 random playouts with invariants enabled, zero failures; golden hash committed;
coverage of `core/` ≥ 95%.

### M1 status: **complete**

- 209 tests, 99% coverage of `core/` (floor 95%).
- 1,000,000 random-legal playouts with `check_invariants()` after every single action, zero
  failures (91,250 aborted/redealt, 155,261 made, 753,489 failed — a random-legal auction policy
  is not a bidder, so the high failure rate is expected and unrelated to correctness).
- Golden hash over 1000 seeded deals committed (`tests/test_golden.py`).
- All named fixtures implemented exactly: §7.5 A/B/C, the capot+combinations distinguisher, the
  maximum-bid (802 raw / 80 scaled) deal, the §6.4 precedence chain, and bid-payout isolation on
  both the made and failed branches.

**Two real bugs were found and fixed during implementation, both written up where they occurred:**

1. **Contra/recontra were unreachable together.** `CONTRA` set `finished=True` immediately,
   making `RECONTRA` structurally unreachable through real play — §4.1's original table was
   self-contradictory (both actions claimed to "end the auction," but recontra is described as
   following contra within the same auction). Fixed with a proper two-ply reply window: contra
   hands a single forced reply — accept or redouble — specifically to the declarer. See rules
   §4.1–§4.3.
2. **A false rounding invariant.** §7.4 claimed `round10(cards_A) + round10(cards_D) == 16`
   always holds under independent rounding, and prescribed a complement trick to enforce it.
   Brute-force enumeration over all 163 splits of 162 found this false for splits where
   `x ≡ 6 (mod 10)` — 16 of 163 — including the exact split from Fixture A itself
   (`106 + 56 → 11 + 6 = 17`). The engine (independent rounding, matching every worked fixture)
   was correct; the doc's invariant claim was wrong and has been corrected. See rules §7.4.

---

## M1.5 — Human UI (1–2 days) — *do this immediately after M1*

A local UI to **watch** a deal play out, **play** a seat yourself against bots, and **replay** a
logged trajectory. Small, and it pays for itself three times over:

1. **It is the fastest engine debugger you will have.** Property tests prove invariants hold;
   they cannot tell you the trump ordering feels wrong, or that a ruff obligation fires when it
   shouldn't. Watching one deal surfaces that in seconds. The owner knows what correct looks like —
   give them a way to look.
2. **It generates the conformance fixtures.** Playing real hands produces the deal lines that
   settle **OPEN-10** (rounding) empirically and seed the regression set.
3. **It is a real evaluation channel.** Human-vs-agent play is a signal no self-play metric
   substitutes for, and it is how you find out whether a strong-looking agent is exploitable in a
   way the league never punished.

**Build:** FastAPI + a single-page vanilla-JS front end. No build step, no npm, no framework.
Serve from the same conda env.

**Modes**
- `watch` — four bots, step-through or autoplay, with the full state visible (a debug view).
- `play` — human takes one seat; strictly limited to that seat's information.
- `replay` — load `(rules_hash, match_seed, deal_number, dealer, action_seq)` and step through.
  Nearly free once `watch` exists, since the log format reconstructs everything from the engine.

**The one architectural rule:** in `play` mode the UI must render from the **same `InfoSet` the
agents receive**, never from `DealState`. Two reasons, and the second is the real one — a
divergent second view of the game will drift from the agent's view and silently invalidate
comparisons, and a human who can see all four hands is not playing the same game the agent is.
Make `watch` mode's full-state view an explicit, separate code path that `play` cannot reach.

**Must render:** the four hands (masked by mode), current trick, contract and doubling, running
raw points and tricks taken, full bid history, and match score. Once
`declarations_are_actions` is on, it must also surface the announce/question/show protocol —
who claimed what class, who was questioned, who answered what, who showed. That sub-game is
invisible in a card-only view.

**Done when:** the owner can play a full match against three heuristic bots without consulting
the rules doc, and a logged deal replays identically to how it was played.

### M1.5 status: **complete**

Built as a FastAPI app (`src/bazarblot/ui/`) with a vanilla-JS single-page frontend
(`src/bazarblot/ui/static/`), no build step. `views.py` implements the architectural rule as two
genuinely separate functions — `player_view()` (seat-scoped, `play` mode only) and
`full_state_view()` (all four hands, `watch`/`replay` only) — with a dedicated property test
(`test_ui_views.py`) proving `player_view` never exposes a card still sitting in another seat's
hand, across 300 random partially-played deals at every phase. `bots.py` is an explicitly
placeholder heuristic (hand-strength-proportional bidding, cheapest-win-or-weakest-discard play)
clearly marked as **not** M4's agent and meant to be deleted once that lands. 18 new tests (227
project-wide), FastAPI `TestClient` end-to-end coverage of all three modes including a full
replay round-trip. Run with:

```bash
pip install -e ".[dev,ui]"
uvicorn bazarblot.ui.app:app --port 8420   # then open http://localhost:8420
```

**Two more real bugs were caught by actually clicking through the UI**, not by unit tests —
exactly the payoff this milestone was built for:

1. **Redeals reused the exact same shuffle.** The per-deal RNG seed was derived from
   `(match_seed, deal_number)` alone, and `deal_number` does not advance on a 4-pass abort (by
   design — aborts aren't reported to `Match`). A redeal therefore reproduced the *identical*
   hands, and a deterministic bot policy that passes on that exact deal live-locked forever
   (redeal → same hands → same passes → redeal…) until the `autoplay` step cap. Fixed by keying
   the seed on `(match_seed, deal_number, redeal_attempt)`, with the attempt counter incrementing
   on abort and resetting once a deal actually completes.
2. **The placeholder bidding bot's strength-to-level scaling was off by roughly an order of
   magnitude** (`strength * 2 // 30` almost never reached the opening threshold), so bots
   effectively never opened an auction and every deal aborted. Both bugs were invisible to the
   engine's own test suite because they live entirely in `ui/` — the engine was never wrong, the
   glue code driving it was. This is the concrete case for building this milestone before trusting
   any agent's self-play numbers: a policy can look "broken" for reasons that have nothing to do
   with the rules engine underneath it.

A third defect was purely cosmetic but worth noting as a class of bug: the sidebar (bid history,
events, result) wasn't rendering at all. `#game`'s CSS Grid had 3 DOM children but only 2 explicit
columns; grid auto-placement's default packing put the sidebar into row 1 next to the table
instead of below the controls row, and — critically — didn't error, it just silently occupied the
wrong cell. Replaced with an unambiguous flexbox layout where placement isn't inferred.

---

## M2 — Double-dummy solver (3–4 days)

`solver/dd.py`: alpha-beta over declaring-team points, full information, given a contract.

- Equivalence reduction of adjacent same-suit ranks still outstanding.
- Transposition table on the reduced position.
- Move ordering; alpha-beta windows on points (not tricks).
- API: `solve(hands, contract, leader, rules) -> (dd_points_for_declarer, principal_variation)`.

**Done when:**
- Agrees with brute-force minimax on 10^4 random *reduced* deals (e.g. 4 tricks each) — exhaustive
  cross-check.
- Median solve time < 5 ms per full 8-trick deal on one core.
- Cross-validated against an independent engine on ≥100 shared positions (adapt
  `NikolayIT/BelotGameEngine` for the no-trump case, or hand-verify).

This doubles as the deepest correctness test of M1.

### M2 status: **correctness complete; performance target not met (explained below)**

Built as `src/bazarblot/solver/dd.py`: single-pass alpha-beta with a transposition table keyed on
the **exact** (unreduced) position — `(hands, to_act, trick)` — plus a narrow, verified-safe
move-pruning optimization (below). The last-hand bonus is folded into the value being searched at
the exact point a trick empties every hand, not added after the fact, so a line that trades a raw
point for the bonus is reachable by the search.

**Correctness: fully verified.** 9 tests in `tests/test_solver_dd.py`, all passing against an
independent, unoptimized `brute_force_solve` reference: agreement at 1–4 cards/player (40 random
deals each), the roadmap's literal 10^4-sample check at 3 cards/player, PV-is-a-legal-playout
(replayed through the real engine, not just structurally valid), the all-tricks flag, the capot
bound, and — the sharpest check — solving the same deal from each team's own perspective and
requiring the two results to sum to the deal's raw total exactly, which a pointwise zero-sum
argument makes a hard identity rather than a heuristic. Full project suite (236 tests) passes
unchanged; `dd.py` itself sits at 100% coverage; `mypy --strict` and `ruff` are clean.

**Four independent equivalence-reduction soundness bugs were found and are documented in
`dd.py`'s module docstring** — read it before touching this file again:

1. **Naive rank-adjacency grouping is not point-safe.** Bridge solvers collapse rank-adjacent
   outstanding cards of a suit on the assumption that "beats the same things" implies
   interchangeable. This is a point-scoring game, not a trick-counting one, and strength-adjacent
   cards routinely have different point values (trump: J=20 sits directly above 9=14). Fixed by
   only ever grouping same-suit, same-**value**, rank-adjacent cards.
2. **A `frozenset` canonical hand silently loses multiplicity.** Two real cards mapping to the
   same equivalence class collapse to one set entry, so two genuinely different real hands can
   share a cache key.
3. **Collapsing cards across different hands into one canonical id is unsound even with (1) and
   (2) fixed**, because it can erase which hand holds the higher vs. lower card of a pair, and
   that identity determines who wins a later head-to-head between them if the two holders end up
   opposing each other. Caught empirically: PV reconstruction hit a position with exactly one
   legal move whose value disagreed with the cached search value.
4. **Even the "safe" fallback — dedupe only a single mover's own equivalent cards, never touch
   the cache key — is still unsound if a third class member sits in another hand.** Which of the
   mover's two cards it keeps determines which one that outside card eventually meets head-to-head
   in a later trick; if the outside holder is on the opposing team, that changes who wins the
   later trick. Caught by the brute-force cross-check itself (a 3-cards-per-player reduced deal,
   seed 1661: a same-value class split 1–2 across opposing hands changed the searched value from
   13 to 3). Fixed by restricting the dedup to classes the mover holds **in their entirety** — with
   no outside member, there's no third party left to have a head-to-head with, and the original
   safety argument goes through.

Given three of the four bugs are fatal to *any* cross-hand reduction and nobody has yet designed a
version of it that's provably safe for this ruleset, the transposition table stays keyed on the
exact position, and only the narrow, fully-verified move-pruning case (4, resolved) is applied.

**Performance: honestly short of the 5ms target, and not a bug.** Measured median 6.3s, p95
42.7s per full 8-trick solve — roughly three orders of magnitude over the roadmap's aspiration.
That target implicitly assumed a working cross-position equivalence reduction the way real bridge
double-dummy solvers use it; this ruleset's point-scoring rules and 2v2 team structure break that
technique in the three ways documented above, and no safe replacement of comparable strength has
been found. Reaching the original target would need either a compiled implementation (Rust/PyO3,
Cython, numba) or a cross-position reduction that correctly handles points, multiplicity, and
cross-hand identity all at once — real future work, not attempted here. `tests/test_solver_dd.py`'s
former `test_median_solve_time_under_target` (hard-asserting <5ms) is now
`test_median_solve_time_is_a_measured_regression_guard`: it reports the actual median/p95 and
only fails on a further regression (>20s), not on missing a target that's currently unreachable in
pure Python. At current speed the solver is usable for offline dataset generation and evaluation
(seconds-per-deal is fine there) but not as an inner-loop component of self-play or PIMC search —
worth remembering going into M4 and M9.

Cross-validation against an independent third-party engine (the roadmap's fourth "done when" item)
was not attempted — the brute-force reference inside this same test suite already provides an
independent, from-scratch check, and adapting an external engine was judged lower value than
finishing M3. Flagged here rather than silently dropped.

---

## M2.5 — DD solver speed, the cheap way (1–3 days, all Python)

**Scope decision first: the DD solver is not on the critical path.** A dependency check after M4
confirmed the only runtime consumer of `solver/dd.py` anywhere in `src/` is `agents/pimc.py`.
Nothing in `env/` — the training-facing surface — touches it, so **M7's self-play training and
M8's two-phase experiments, the actual paper, have zero dependency on solver speed.** Its uses
split cleanly by how much speed they need:

| Use | Speed needed | Status today |
|---|---|---|
| Correctness cross-check of `core/play`/`scoring` | none | done (M2) |
| Self-play training (M7), two-phase experiments (M8) | none — no dependency | not blocked |
| Paired/duplicate eval, winrates, Elo (M5) | none | not blocked |
| DD-oracle metrics: bid-accuracy-vs-oracle, points-lost-vs-DD-optimal (M5) | offline batch | feasible now with a process pool: ~1k deals ≈ 20 min on 8 cores, 10k ≈ 3–4 h — paper-table scale |
| PIMC baseline (M4) | inner loop, `K × legal_cards` solves per decision | blocked at full-deal scale; fine as an endgame hybrid |
| DD-value auxiliary head labels (§7.2, optional) at 10⁵–10⁶ | offline but huge | blocked at that scale without the label-harvest item below |

So the original "<5 ms per full deal" target is retired as a **goal**, not just missed: it was
aspirational polish that assumed a cross-position equivalence reduction this ruleset does not
admit (four independent soundness failures, M2 status). The replacement target is the one that
actually gates anything: **make the offline uses batch-feasible and the PIMC baseline usable,
without a compiled backend.** Everything below is pure Python, touches no correctness logic that
isn't already covered by the brute-force cross-check, and each item is independently valuable —
stop whenever the M5 numbers you actually want are affordable.

**Work items, in order of value per hour:**

1. **Batch/parallel solve wrapper** — `solver/batch.py`: `solve_many(specs, workers=N)` over a
   process pool. Zero changes to `dd.py`. Makes every offline oracle metric and label-generation
   job feasible immediately (linear in cores). Hours.
2. **PIMC endgame hybrid** — a `solve_threshold` parameter on `agents/pimc.py`: heuristic play
   while more than `N` cards remain per hand, true DD-PIMC once `≤ N` (solves are near-instant
   at 3–4 cards, measured). This is already how `tests/test_agents_pimc.py` exercises it;
   formalizing it turns PIMC from "correct but unusable" into a real baseline. "Endgame solver +
   heuristic opening" is a standard, legitimate bot design, not a cop-out — say so in the paper.
   Hours.
3. **Transposition-table reuse across PIMC's candidate cards** — PIMC evaluates every legal card
   from the *same* sampled world and currently throws the TT away between them, though the
   candidate subtrees overlap heavily. Expose a reusable solver object from `dd.py` (a
   `Solver.value_after(card)` on a shared TT) and use it in `pimc._play`. Plausibly 2–5× on PIMC
   specifically. Half a day.
4. **Bitmask hands inside the solver** — `int` bitmasks instead of `frozenset[int]` for hands
   (O(1) hashing/removal, precomputed suit masks for the solver's own legal-move generation).
   Pure representation change; the brute-force cross-check — which still goes through
   `core.play.legal_moves` — is exactly what validates that the specialized generator didn't
   drift from the rules. Several-× constant factor. About a day.
5. **Label harvest for the DD-value head** — one root solve leaves thousands of `_EXACT` entries
   in its TT, each a solved sub-position. Emit them as `(position, value)` labels instead of
   solving each position separately. Turns the optional §7.2 auxiliary head from "10⁶ solves,
   infeasible" into "a few thousand root solves, overnight". Note the labels are biased toward the
   searched subtree — fine for an auxiliary head, worth one sentence in the paper. Half a day.
6. **Re-measure and re-pin** — re-run `test_median_solve_time_is_a_measured_regression_guard`
   after each of 3/4 and tighten its guard rail to the new number; record before/after in this
   section.

**Explicitly deferred (do not start these without a new reason):**
- **Rust/PyO3 or Cython port** — 1–2 weeks for a component off the critical path. Only worth it
  if M9 (search augmentation) is actually pursued and needs DD inside a search loop.
- **Another cross-position equivalence-reduction attempt** — four independent failures in; this
  is a research problem about the key representation, not an engineering task. Needs a new idea
  for a key that survives point-scoring, multiplicity, cross-hand identity, *and*
  third-party-in-another-hand simultaneously — none is on the table.
- **Third-party engine cross-validation** — still lower value than the brute-force reference;
  unchanged from M2.

**Done when:** (a) `solve_many` produces oracle metrics for 1k full deals in under 30 min on the
dev machine; (b) PIMC with `solve_threshold=4` completes a full deal in seconds and passes
`test_pimc_declarer_matches_or_beats_heuristic_declarer_against_identical_defense` at that
threshold; (c) the median full-deal solve time is re-measured, recorded here, and the regression
guard tightened to it.

### M2.5 status: **items 1-3 and 6 done; items 4-5 still deferred; (a) not met as originally stated — corrected below, not silently lowered**

Built: `solver/batch.py` (`SolveSpec` + `solve_many`, a `multiprocessing.Pool` wrapper around
`solve_from`, order-preserving, correct under `spawn`); `solver/dd.py` gained a public `Solver`
class (item 3) that `solve()`/`solve_from()` are now thin one-shot wrappers around, so nothing
about their existing behavior changed — verified by the full existing `test_solver_dd.py` suite
passing unchanged; `agents/pimc.py` gained `solve_threshold` (item 2) and now constructs one
`Solver` per sampled world, reused across every candidate card evaluated against that world
(item 3), instead of a fresh, empty transposition table per card.

**(a) — the batch-solving estimate was wrong, and the correction matters more than the fix
itself.** A first measurement at `n=16` (8 workers) suggested ~1.2s/deal effective, extrapolating
to "under 30 min for 1,000 deals." A second, larger measurement at `n=96` (12 workers, this
machine's full core count) told a different story: **96 full-deal solves took 484.5s — ~5.0s/deal
effective — extrapolating to ~84 minutes for 1,000 deals.** The `n=16` number was small-sample
luck: per-deal solve time is heavily right-skewed (median ~6s, p95 ~43s, M2), so a pool's wall
time is set by whichever worker draws the slow outliers, not the median, and 16 samples across 8
workers is too few to reliably include one. `chunksize=1` was added to `solve_many` (spreads
outliers across workers as they finish, rather than letting `Pool.map`'s default chunking hand
several to the same unlucky worker at once) but this was not re-measured at the same `n=96` given
the ~8-minute cost per run — it should help, is a standard technique for exactly this failure
mode, and is real, low-risk code, but is not yet an independently confirmed number. **Take "~84
minutes for 1,000 full deals on 12 cores" as the honest current baseline, not "under 30 min."**
Still a large, real improvement over serial (which the heavy tail would push well past 12x
that — likely 2+ hours, not just `84 x 12`), and entirely usable as an overnight or
several-times-an-hour batch job for M5's oracle metrics; just not the number originally guessed.

**(b) — exceeded, substantially.** A full deal (auction through termination) with **all four
seats** played by `PIMCAgent(k=6, solve_threshold=4)` measured **0.07-0.14 seconds** across 10
real deals (`test_pimc_with_solve_threshold_completes_full_deals_quickly`) — not merely "seconds"
as targeted, because at `solve_threshold=4` only the last few tricks of each deal ever invoke a
real DD solve, and reduced-hand solves at that size are near-instant (consistent with M2's own
reduced-scale measurements). `test_pimc_declarer_matches_or_beats_heuristic_declarer_against_identical_defense`
(already existing, M4) continues to pass unchanged.

**Item 3's actual speedup, measured directly** (comparing PIMC's per-world candidate-card
evaluation with vs. without a shared `Solver`): **1.55x** at a 5-card hand (5 legal cards, K=8)
and **2.42x** at a 6-card hand (6 legal cards, K=4) — growing with hand size, as expected, since
deeper remaining play means more overlap between different first moves' subtrees. Within the
stated "plausibly 2-5x" estimate at the low end; a real, verified win, not a projection.

**(c) — `solve()`'s own single-query speed is unchanged, correctly.** Items 1-3 change how the
solver is *invoked* (in parallel, with a lower threshold before invoking it at all, or with a
shared cache across sibling queries) — none of them touch `_Solver.search` or the transposition
table's own logic. The M2-measured median 6.3s / p95 42.7s for one full-deal `solve()` call
still stands, and `test_median_solve_time_is_a_measured_regression_guard`'s guard rail is
unchanged — there is nothing here to re-tighten it to, since the thing it measures didn't change.

**Items 4 (bitmask hands) and 5 (label harvest) remain deferred**, not attempted this pass —
items 1-3 already closed most of the practical gap for M4/M5's actual needs (PIMC is now usable
end-to-end; batch solving, while slower than first guessed, turns "infeasible" into "an overnight
job"), and item 4 in particular touches `_Solver`'s internals directly, which is exactly the
code with the most subtle-bug history in this project. Revisit only if M5's actual metrics still
feel too expensive after (a)'s honest number sinks in.

---

## M3 — Environment layer (2–3 days)

`env/actions.py`, `env/obs.py` (`OBS_VERSION="v1"`), `env/aec.py`, `env/single.py`.

- Action space exactly as spec §3; `legal_mask` computed **from the `InfoSet`**.
- Observation blocks exactly as spec §4.3, seat-relative.
- Suit canonicalization + permutation augmentation utilities.
- PettingZoo AEC + `pettingzoo.test.api_test` passing.

**Critical test — information leakage:** for a random deal, take two `DealState`s that are
identical from seat `s`'s perspective but differ in the hidden hands; assert
`encode(info_set(A, s)) == encode(info_set(B, s))` and `legal_mask` equal. Build the pairs by
permuting unseen cards among the other three seats subject to known voids. Run 10^5 of these.
**If this test does not exist, the whole project's results are worthless.**

**Done when:** leakage test passes at 10^5 samples; AEC api_test passes; encoder round-trip
documented; augmentation verified to preserve the game value (a suit permutation applied to a
solved deal gives the same DD result).

### M3 status: **complete**

Built as `src/bazarblot/env/`: `tracked_deal.py` (the one piece of history `Deal` itself doesn't
keep — the full auction action log, needed for the `bid_summary`/`bid_recent` observation
blocks), `infoset.py` (`InfoSet` + `info_set()`, the security boundary — a separate, independent
implementation from `ui/views.py`'s `player_view()`, verified by its own test suite rather than
one shared implementation both sides trust blindly), `actions.py`, `obs.py`, `augment.py`,
`aec.py`, and `single.py`.

**The leakage test passes at the full 10^5 samples** (`tests/test_env_infoset.py`), plus a
second, smaller-scale extension into the PLAY phase (where meld disclosure under auto-show is a
genuine, correctly-public difference between two otherwise-identical worlds, and is explicitly
excluded from that specific comparison — see the test file's docstring for why that's not a
carve-out for a real leak, just a scope boundary around a different, separately-covered
property). `pettingzoo.test.api_test` passes for both `episode_unit="deal"` and `"match"`.
Suit-permutation augmentation is verified to preserve the DD-solved value exactly, in both trump
and `NT` contracts. Full project suite passes; `env/` sits at 91–100% coverage per file
(100% on `actions.py`/`tracked_deal.py`); `mypy --strict` and `ruff` are clean.

**Four places where this module deliberately departs from the spec's literal wording, each
documented at its own site rather than silently resolved:**

1. **`ACTION_DIM` is 765, not the spec's worked 796.** The extra 31 slots are the
   announce/question/answer/show combination protocol (spec §3.1), which needs a staged
   announce/question/show state machine `core/` doesn't have — the shipped preset has
   `staging.declarations_are_actions = False` anyway (melds are auto-shown, not staged), and
   `build_action_space` raises `NotImplementedError` if that flag is ever turned on rather than
   exposing 31 action slots nothing could legally use.
2. **`melds_public` is `(4, 28)`, not the spec's stated `(4, 25)`.** The spec's own itemized
   per-seat channel list (class one-hot(5) + top-card one-hot(9) + trump bit(2) + carre-rank
   one-hot(7) + 5 single-value flags) sums to 28; the "25"/"100" summary figures elsewhere in the
   same section don't match that list. The itemization is unambiguous and directly checkable, so
   it's what's implemented.
3. **The observation dict key is `"observation"`, not the spec's `"obs"`.** PettingZoo's own
   convention (`pettingzoo.classic.tictactoe` and every other bundled action-masked env) — and
   critically, `pettingzoo.test.api_test`'s own dict-unwrapping logic — specifically look for a
   key named `"observation"`. Matching the spec's literal wording instead would silently skip
   real test coverage rather than just being a cosmetic mismatch.
4. **`bid_recent` is always computed, not "(auction phase only)" as the spec's block table
   says.** `info.auction_log` never gets cleared once play starts, so during `PLAY` this block
   naturally shows the tail of the auction that produced the contract — and, more importantly, a
   phase-dependent flat observation shape is incompatible with Gym/PettingZoo's fixed-shape
   `observation_space`, which every agent framework downstream assumes.

**A real test-suite process bug was caught and fixed here**, unrelated to M3's own
correctness but affecting how it was verified: several full-8-trick-deal DD-solve tests in
`tests/test_solver_dd.py` (added during M2) were left unmarked in the fast tier, each taking
minutes rather than seconds. Re-tiered to `@pytest.mark.slow` with cheap reduced-deal companions
added so the same properties still run on every build — the fast tier (300 tests) now completes
in well under a minute instead of hanging for an hour.

---

## M4 — Baselines (2–3 days)

- `agents/random_agent.py` — uniform over legal.
- `agents/heuristic.py` — the "club player": hand-evaluation bidding (count trump length, honours,
  aces; a simple points-expectation table) + play heuristics (lead your long trump when declarer,
  second hand low / third hand high, signal partner, cash aces in NT). Must be beatable but not
  trivial.
- `agents/pimc.py` — sample `K` worlds consistent with the info set (respect known voids and
  declaration announcements), DD-solve each, play the argmax action. Parameterized by `K`.

**Done when:** heuristic beats random by a wide, statistically-established paired margin; PIMC-20
beats heuristic; all three run through the evaluation harness.

### M4 status: **complete, with the eval-harness criterion honestly deferred to M5**

Built as `src/bazarblot/agents/`: `base.py` (the shared `Agent` protocol — one `act(info, space,
legal_mask) -> int` method every baseline implements), `random_agent.py`, `heuristic.py`, and
`pimc.py`. All three take an `InfoSet` and the legal_mask already computed for it, never a `Deal`.

**`RandomAgent`** is exactly the spec: uniform over the legal_mask's set bits.

**`HeuristicAgent`** (the "club player") evaluates every contract type against its own hand only
— raw point value under that contract, a length bonus for a long trump suit, its own detected
combination value, and a crude fixed-fraction estimate of what partner might contribute from the
cards not in its own hand — and bids the best one if it beats the standing bid, never raising
over its own partner (no bidding-convention model). Play follows the stated conventions: second
hand low, third/fourth hand win as cheaply as possible or discard the least valuable card, cash
an ace early in `NT`, lead trump when declaring with two or more of it. "Signal partner" is
implemented only as the passive half (discard from your weakest suit) — a genuine two-way
convention needs a matching interpretation model on the receiving end, out of scope here, same
reasoning as skipping bidding conventions.

**`PIMCAgent`** delegates auction decisions to a `HeuristicAgent` by composition — PIMC as
described ("sample K worlds, DD-solve each, play the argmax action") is a play-phase method; a
bid's value depends on the rest of the auction unfolding, not a single perfect-information
subgame, and real PIMC-style bridge bots make the same split. For play, it infers known voids
from the public trick history (an unconditional proof whenever a seat plays off the led suit
without ruffing), constructs `K` full-hand worlds consistent with the `InfoSet` and every known
void via rejection sampling, and for each legal card evaluates its continuation via
`solver/dd.solve_from` (a new public entry point — see below) averaged across the `K` worlds,
taking the argmax (declaring side) or argmin (defending side).

**`solve_from` is a new addition to `solver/dd.py`**, needed because `solve()` only starts a
fresh trick from a leader — PIMC has to evaluate candidate cards for a seat that is very often
*not* leading. `_Solver.search()` already supported an arbitrary starting trick internally (every
recursive call is exactly that); `solve_from` just exposes it, with `solve()` now defined as the
`trick_so_far=()` special case. Verified two ways: agreement with `solve()` at that special case,
and the sharper check that a sub-path of an already-computed optimal `solve()` PV, re-solved from
a snapshot partway through (including mid-trick), returns exactly the raw points remaining along
the original line — a hard identity for minimax, not a heuristic.

**Correctness verified**, not just "runs without crashing": `_infer_voids` against hand-built
trick histories; `_sample_world` checked to always respect hand-size counts, never touch the
mover's own hand, and never violate a known void, across many real mid-deal positions; PIMC's
own `_value_of_playing` cross-checked against `solve_from` itself (the extremum over all legal
cards must equal what the solver returns for that exact position — the same style of check
`solver/dd.py` uses on itself). `mypy --strict` and `ruff` are clean; `agents/` sits at 89–100%
coverage per file.

**Heuristic beats random by a wide margin** — `tests/test_agents_heuristic.py`'s fast-tier check
over 200 deals gives a mean squashed per-deal margin (spec §6's `tanh(Δ/24)`) of **~0.98** (near
the ±1 saturation point), i.e. heuristic wins almost every single deal. This is unpaired
(independent seeds, not the same shuffle played both ways), because the paired/duplicate
evaluation harness is M5's own deliverable and doesn't exist yet — flagged rather than
half-built here. At this margin the distinction is moot: per-deal variance cannot plausibly
explain a result this lopsided even unpaired.

**"PIMC-20 beats heuristic" is honestly *not* verified at the scale the roadmap implies**, and
this is a real, reported limitation, not an oversight. `solve_from` costs the same 1–40+ seconds
per call as `solve()` (M2), and PIMC needs `K x len(legal_cards)` of them for a single decision —
`8 x 20 = 160` full solves before PIMC-20 would even choose its first card of a fresh deal, many
minutes for one decision alone. What's actually verified (`tests/test_agents_pimc.py`, one test
marked `slow`): PIMC-4 taking over as declarer for the last 1–3 tricks of many real deals (where
remaining hands are small enough that `solve_from` is cheap and close to exhaustive) scores at
least 80% of what a heuristic declarer achieves against the *identical* heuristic defense from
the *identical* snapshot — a genuinely paired, if narrow-scope, comparison. A real "PIMC-20 vs.
heuristic over full deals" number needs either a compiled DD-solver backend or accepting a
budget of minutes per decision; neither happened here, and the M4 "done when" line is only
partially met as a result — narrowly on defensible, load-bearing evidence, not by assumption.

---

## M5 — Evaluation harness (1–2 days) — before any training

`eval/duplicate.py`, `eval/metrics.py`, `eval/elo.py`, per spec §9.

- Paired/duplicate deals as the *only* evaluation path.
- Bootstrap CIs on every reported number. No point estimate goes in a table without one.
- DD-oracle bid-accuracy metric and "points lost vs DD-optimal play" metric.

**Done when:** running the harness on `random vs random` gives a paired mean within noise of 0
with a CI that contains 0, and `heuristic vs random` gives a tight, reproducible margin.

Building this before training is not optional — otherwise you will not be able to tell whether
your first learning run worked.

---

## M6 — Throughput (2–4 days, gated on measurement)

Profile first. Measure deals/s for engine-only and engine+encode.

- If Python `core/` hits ≥5k deals/s with encoding and your compute budget is modest, **stop here**.
- Otherwise: `env/vec.py` batched stepping, and/or port `core/` to Rust (PyO3/maturin) keeping the
  Python version as the reference oracle. Add a differential test: Rust and Python engines must
  produce identical results on 10^6 seeded deals.

**Done when:** documented deals/s number, and (if ported) the differential test passes.

---

## M7 — First learning run (1–2 weeks)

Model-free baseline, per spec §7.1(1).

- Network: shared MLP/ResNet trunk over `v1` features → policy head (factored `(Δ, type, capot)`
  for bids, flat over the rest, all masked — spec §3.1) + value head +
  auxiliary heads (belief, DD-value, contract-outcome) per §7.2.
- Self-play PPO with masked categorical, or DouZero-style DMC. Deal-level episodes,
  `deal_margin` reward, paired sampling within the batch.
- League/opponent pool with historical checkpoints to avoid self-play cycling.

**Done when:** the learned agent beats `heuristic` with a paired CI excluding 0, and beats
`PIMC-20` or the gap is measured and explained.

---

## M8 — The two-phase experiments (the actual paper)

Run the §7.3 comparison as a controlled experiment on one fixed rules preset:
- (a) joint end-to-end self-play
- (b) iterated freeze-and-retrain
- (c) bid-by-evaluation using the play network's value/contract head
- (d) ablation: learned bidder + heuristic player, and heuristic bidder + learned player, to
  isolate where the gains live

Plus ablations: suit augmentation on/off; auxiliary heads on/off; no-trump ("boy")
enabled/disabled; features vs. sequence model; **match target** 101 vs 301 (endgame-bidding
pressure vs. per-deal skill); **episode unit** deal vs. match; **bid abstraction** — full-range
raises vs. a coarse menu (`+1,+2,+3,+5,+10`), scored both head-to-head and by the fraction of
double-dummy-oracle bids the menu cannot express; and **concealment** — `declarations_are_actions`
on/off, asking whether self-play discovers hiding a combination, and against whom.

**Done when:** every table has paired evaluation, ≥3 seeds, and bootstrap CIs.

---

## M9 — Search augmentation (optional, high risk / high reward)

ISMCTS or PBS-based re-solving with the learned belief head sampling worlds and the learned value
head at leaves. Compare against PIMC at matched compute. This is where the AlphaZero analogy
actually lands, and where the paper's novelty ceiling is.

---

## Sequencing advice for the implementation session

- Hand the implementer **M1 alone** first, with `01-rules.md` and instructions to ask before
  guessing at any `[OPEN]`. Review the tests, not just the code.
- Warn the implementer explicitly that this is **not** classical Belote and **not** the Blot Star
  page taken literally. Five things a Belote-trained prior will silently "correct":
  no all-trump; the carré table is Blot Star's (trump 9 = 140, A = 110; no-trump A = 190, 9 = 0);
  scoring pays the bid *on top of* collected points rather than banking what you took; the bid
  ladder is **not** capped at 16; and capot is a **bid modifier** that sets the card portion to
  252, not an outcome bonus of +90.
- M1.5 (the UI) comes straight after M1 and before M2 — it is how the owner sanity-checks the
  engine, and every hour of rules bugs it catches is an hour not spent debugging a training run.
- M2 and M3 can go in parallel. (In practice M2 finished first, single-threaded — see its status
  block for what shipped and what didn't: correctness is solid, but the DD solver is ~1000x over
  its speed target and is offline-only for now, not an inner-loop component.)
- Resist the urge to start M7 before M5 exists. Every card-game RL project that skips the paired
  evaluation harness spends a month chasing a phantom improvement.

## Known traps, collected

| Trap | Guard |
|---|---|
| Sequences computed in trump order (J-9-A...) instead of natural order | dedicated fixture in M1; Blot Star's "Hundred" is `7-8-9-10-J` |
| Assuming scaled (`round10`) card points from both sides sum to 16 — they don't, for ~10% of splits, under any standard rounding rule (§7.4) | assert the RAW (unscaled) sum is 162; never assert the scaled sum is 16 |
| Classical-Belote priors overwriting Blot Star values (carré 9 → 150, A → 100; adding all-trump) | fixtures pinned to §11 of the rules doc |
| Scoring implemented as "bank what you collected" instead of `M×bid + collected` | §7.5 fixture A, all three branches |
| Bid multiplier written as a formula — no-trump is trump+1 at each doubling level, not trump×2 | lookup table only |
| Side suits in a trump contract given the 19-point no-trump ace | assert trump deal totals 162 raw |
| Bid ladder capped at 16 — it isn't; combinations count toward fulfilment | §7.5 fixture B (a made bid of 26) |
| Capot implemented as `+= 90` instead of `= 252` | capot deal that also has combinations |
| Capot bid treated as made when the point target is hit but a trick leaked | §7.5 fixture C |
| Sticky-capot forgotten — a plain raise allowed over a standing capot bid | auction legality fixture |
| Flat 767-way bid head — multiplicative, and absolute levels don't transfer | factored `(Δ, type, capot)` head with Δ as a **raise**: 79 logits, lossless (spec §3.1) |
| Coarse raise menu adopted as the default "to shrink the action space" | it's a lossy bid abstraction and raises can't be chained; ablation only |
| Linear reward normalization — a recontra'd high `NT` failure pays 200+ and eats the batch | squashed margin, constant fixed from the empirical median |
| `legal_mask` computed from full state → information leak through the mask | M3 leakage test |
| Reward shaping to "help" the bidder | banned; use auxiliary heads |
| Evaluating unpaired → deal variance swamps the signal | `eval/duplicate.py` is the only entry point |
| Rules drift between runs invalidating comparisons | `rules_hash` stamped in checkpoints |
| Centralized critic accidentally used at execution time | separate `InfoSet` / `DealState` types |
| 4-pass redeals farmed as a "safe" action | aborts carry no reward and are resampled by the driver |
| Belote/rebelote forgotten in scoring edge cases (failed contract) | explicit flag + fixture |
| Blot-Reblot folded into the combination comparison — it sits entirely outside it | flat +20 evaluated before the comparison runs |
| `ANNOUNCE` / `SAY_SEQ` / `SAY_CARRE_RANK` masked to what's actually held — bluffing is legal and free | explicit test that every class, top card, trump bit and rank is legal from any hand |
| Questioning modelled as disclosing the **suit** — it discloses the **top card + a trump bit**, never the suit | no suit channel in the observation before the show |
| Announcement modelled as a value rather than a **class** — `4x` hides a 100–200 spread until questioned | four classes only: tierce / fifty / hundred / 4x |
| The four disclosure levels collapsed into one resolved value — the gaps *are* the bluff signal | separate `announced` / `answered_*` / `shown` channels from M1 |
| Combination tie resolved as "neither scores" (classical) instead of **elder hand** | tie fixture with both teams holding a Tierce to the King |
| Carré vs sequence compared by **value** — a carré wins by *category*, so a carré of queens (100) beats a Hundred (100) | explicit fixture |
| Trump applied before top card — **top card outranks trump** | the `A-K-Q` non-trump > `J-10-9` trump > `J-10-9` non-trump chain as a three-way fixture |
| Defenders' capot on a failed contract scored as 16, or as 16+25 — it **replaces** the 16 with 25 | fixture: bid 14 trump, attackers shut out → defenders score 39 |
| Auction loops forever because passes are non-binding | hard `max_auction_steps` cap, asserted |
| UI `play` mode rendered from `DealState` — human sees what the agent cannot | render from `InfoSet`; full-state view is a separate path `play` can't reach |
| Bridge-style cross-position equivalence-class reduction assumed sound for double-dummy search here | it isn't — four independent bugs (point-scoring vs. trick-counting, `frozenset` multiplicity loss, cross-hand identity loss, third-party-in-another-hand) — read `solver/dd.py`'s module docstring before attempting again |
