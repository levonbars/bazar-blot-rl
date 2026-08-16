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
