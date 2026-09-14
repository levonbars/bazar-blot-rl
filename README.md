# Deep RL for Bazar Blot (Armenian Bazaar Belote)

Research project: learning to play **Bazar Blot** — a 32-card, 2-vs-2, imperfect-information
trick-taking game with a bidding phase — with deep reinforcement learning and self-play.

## Why this game

Most solved/attacked card games isolate one difficulty. Bazar Blot stacks several:

- **Imperfect information.** 24 of 32 cards are hidden at the start of play. AlphaZero's PUCT is
  unsound here; this is the framing of the work, not a footnote.
- **Two coupled phases.** An auction (*bazar*) fixes the contract, trump, and a numeric commitment;
  then 8 tricks decide whether the commitment is met. The optimal bidding policy is *defined by*
  the strength of the play policy, and vice versa. That circular dependency is the central
  research question here, and it has no clean analogue in Chess/Go/Poker.
- **Cooperative and adversarial simultaneously.** Your partner is a separate agent with different
  private information, whom you cannot talk to except through the legal actions themselves —
  card play and bids carry conventional signalling. Team play must emerge without communication
  channels, while two opponents actively work against it.
- **A bluffing sub-game inside a trick-taking game.** Combination announcements are costless and
  unverified until the show step, so they are cheap talk with partial commitment — closer to
  poker's bet-as-signal than to Bridge's conventions — and they are coupled to the play phase
  through the hand information a show reveals.
- **Small enough to be tractable.** 8 tricks, ≤8 legal moves per node. The perfect-information
  play phase is *exactly solvable* in milliseconds, which gives an oracle for evaluation,
  distillation, and a strong non-learned baseline (PIMC) — a luxury Poker and Mahjong don't have.

## Documents

| Doc | Contents |
|---|---|
| [`docs/01-rules.md`](docs/01-rules.md) | Formal rules spec against the **Blot Star** ruleset. Card values, auction, combinations, scoring, config flags, and the open questions. |
| [`docs/02-environment-spec.md`](docs/02-environment-spec.md) | Environment design: layering, state/info-set model, action space, observation encoding (`v1`), reward, algorithm candidates, evaluation harness. |
| [`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) | M0–M9 milestones with explicit "done when" criteria, and the collected list of traps. |
| [`presets/blotstar.yaml`](presets/blotstar.yaml) | The ruleset as machine-readable config. Every value tagged with its provenance; hashed into `rules_hash`. |

## Ruleset

The target ruleset is **[Blot Star](https://blotstar.com/bazar-blot-rules/)**, corrected and
extended by the project owner where the published page is a simplification. It is not classical
Belote, and five differences matter enough that a Belote-trained intuition will silently "correct"
them:

- **Two contract families only** — a trump suit, or no-trump ("boy"). No all-trump.
- **Four-of-a-kind values are Blot Star's own** — in a trump contract nines are 140 (not 150) and
  aces are 110 (not 100); under no-trump aces are 190 and nines are worth nothing.
- **Scoring pays the announced bid on top of collected points** (`M × bid + collected + bonuses`),
  and hands the defenders the entire deal plus the bid on failure. No-trump doubles the bid term;
  contra and recontra scale it further on a lookup table (trump 1/2/4, no-trump 2/3/5).
- **The bidding ladder is not capped at the deal's 16 points.** Combination bonuses count toward
  fulfilment, so a hand with a carré of jacks and a Fifty is holding 25 points before a card is
  played. The ladder runs to **80** — capot 25 plus all four scoring carrés (20 + 14 + 11 + 10).
- **Capot is a bid modifier, not just an outcome.** `26` and `26 Capot` are different bids; once
  capot is bid every higher bid must also be capot; and a capot bid needs *both* the point target
  and all 8 tricks — hitting 27 points while leaking one trick still loses.
- **Combinations are claimed through a staged announce/question/show protocol with free bluffing.**
  At trick 1 you announce only a *class* — Tierce, Fifty, Hundred, or 4x — where 4x hides anything
  from 100 to 200. An opponent holding a combination may then question you — you disclose your
  sequence's top card and whether it's trump, or a carré's rank, but never the suit — and you must
  answer, though you may lie. Only showing the cards at trick 2 actually scores. If the leading
  claimant declines to show, a runner-up who also announced can claim instead. Showing leaks up to
  five of your eight cards right before the play phase, so the real trade is points for
  information — at three separate points where the agent can stop.

The scoring model is the interesting part for RL: overbidding is *rewarded* when it comes off and
punished hard when it doesn't, which makes the auction a far sharper decision problem than
classical Belote's "bank what you take". The capot flag adds a second, independent failure mode on
top of the point target — the sharpest single decision in the game.

## Setup

```bash
conda env create -f environment.yml && conda activate bazarblot && pip install -e ".[dev,ui]"
```

```bash
pytest -q && ruff check . && mypy
```

### Watch / play / replay a deal

```bash
uvicorn bazarblot.ui.app:app --port 8420   # then open http://localhost:8420
```

Three modes: **watch** four bots play each other with full state visible, **play** one seat
yourself against three bots (only your seat's information is ever sent to the browser), or
**replay** a logged deal (paste the JSON from "Copy replay log" after a deal finishes) and step
through it. See [`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) M1.5 for
the design (why `play` and `watch` are two genuinely separate code paths, not one function with a
"reveal everything" flag) and the bugs building it caught.

## Status

**M0 through M5 complete.**

The rules engine (`src/bazarblot/core/`) implements auction, trick play, combination detection
and scoring end to end. 236 tests, 99–100% coverage of `core/` and `solver/` (floor is 95%), a
golden-hash regression fixture over 1000 seeded deals, and a 1,000,000-deal random-legal-playout
validation with invariants enabled and zero failures. A local FastAPI UI (`src/bazarblot/ui/`)
can watch, play, and replay deals against a placeholder bot.

Four real bugs were caught and fixed during implementation, not just during design — two in the
engine, two only surfaced by actually clicking through the UI:
- **Contra/recontra were unreachable together.** The original auction logic set `finished=True`
  immediately on `CONTRA`, which made `RECONTRA` structurally impossible to reach through play —
  the rules doc's own §4.1 table was self-contradictory (both actions claimed to "end the
  auction," but recontra is described as coming *after* contra in the same auction). Fixed by
  giving contra a proper two-ply reply window (§4.3): it hands a forced single reply — accept or
  redouble — specifically to the declarer, not their partner or a new round of bidding.
- **A false rounding invariant in the rules doc.** An earlier draft of §7.4 claimed that scaling
  both sides' card points independently always sums to 16 (`round10(cards_A) + round10(cards_D)
  == 16`), and prescribed a "complement" trick to guarantee it. Brute-force enumeration found this
  is mathematically false for ~10% of splits — including the exact split from the doc's own
  Fixture A (`106 + 56 → 11 + 6 = 17`, not 16). The engine matches the doc's worked fixtures
  (independent rounding, no complement), so the code was right and the doc's claim was corrected.
- **Redeals reused the exact same shuffle,** since the per-deal seed was keyed only on
  `(match_seed, deal_number)` and `deal_number` doesn't advance on a 4-pass abort. A deterministic
  bot that passes on one exact hand would redeal into that *same* hand forever. Fixed by keying
  the seed on `(match_seed, deal_number, redeal_attempt)`.
- **The placeholder bot's bidding threshold was off by about 10x**, so it essentially never
  opened an auction — every deal aborted. Both UI-layer bugs were invisible to the engine's own
  (extensive) test suite, because the engine was never wrong — the code driving it was. That's
  the concrete argument for building M1.5 before trusting any agent's self-play numbers later.

All four corrections are written up where they occurred — [`docs/01-rules.md`](docs/01-rules.md)
§4.1–§4.3 and §7.4, [`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) M1.5
— not just fixed silently.

Two rules questions remain open (auction termination after three passes, and half-up vs half-down
rounding), both behind config flags in `presets/blotstar.yaml` and neither blocking.

**M2** (`src/bazarblot/solver/`) adds a double-dummy solver — an alpha-beta oracle that, given all
four hands, computes the maximum raw points the declaring side can force under optimal play from
both sides. Correctness is fully verified: 9 tests cross-check it against an independent
brute-force reference (reduced-deal agreement, a literal 10,000-sample bulk check, PV replay
through the real engine, and a zero-sum identity — solving the same deal from each team's own
perspective must sum to the deal total exactly). Chasing this surfaced **four independent
soundness bugs** in the classic bridge-solver "equivalence class" speedup, all written up in
[`solver/dd.py`](src/bazarblot/solver/dd.py)'s module docstring — the short version is that
collapsing rank-adjacent cards into one cache-key class, standard practice in trick-counting
bridge solvers, is unsound in a *point*-scoring 2v2 game in three different ways, and unsound a
fourth way even in a scoped-down "same hand only" form. The transposition table therefore stays
keyed on the exact position, and performance is honest about the cost: **median 6.3s, p95 42.7s**
per full 8-trick solve, roughly 1000x over the original 5ms aspiration (which assumed the
now-abandoned reduction). That makes the solver solid for offline dataset generation and
evaluation, but not yet fast enough to sit in a self-play or PIMC inner loop — closing that gap is
future work (a compiled backend, or a correctly-designed cross-position reduction nobody has built
for this ruleset yet).

**M2.5** revisits that gap the cheap way — a dependency check found `agents/pimc.py` is the DD
solver's *only* runtime consumer (self-play training, M7/M8, has zero dependency on it), so the
original 5ms target is retired in favor of making the actual downstream uses affordable. Three
changes landed: `solver/batch.py` parallelizes solving across processes; `solver/dd.py` gained a
public `Solver` class so repeated queries against the same contract/team can share one
transposition table instead of starting fresh each time; and `PIMCAgent` gained
`solve_threshold`, so it plays large hands with a cheap heuristic and only DD-solves once few
enough cards remain. The last one is the standout result: a **full deal, all four seats, PIMC
with `solve_threshold=4`, completes in 0.07-0.14 seconds** — not "seconds" as targeted, because
near-endgame solves are cheap. The transposition-table sharing measured a real 1.6-2.4x speedup
on PIMC's own candidate-card evaluation. The batch-solving estimate, though, needed an honest
correction: a small first measurement suggested ~30 minutes for 1,000 full deals; a larger,
more representative one (96 deals, 12 workers) found the per-deal cost distribution is heavy-
tailed enough that the real number is **closer to 84 minutes** — still a large win over serial,
just not the number first guessed. Full account in `docs/03-implementation-roadmap.md`'s M2.5
status section, including why the first estimate was wrong, not just what the second one says.

**M3** (`src/bazarblot/env/`) adds the RL-facing environment layer: an `InfoSet` security
boundary derived independently from the UI's own (`ui/views.py` and `env/infoset.py` are two
separate implementations of "what can seat X see," each with its own test suite), a flat
phase-masked action space, a `v1` tensor observation encoder, suit-permutation data augmentation,
and both a PettingZoo AEC multi-agent env and a Gym single-agent wrapper. The load-bearing check —
the leakage test — passes at the full 10^5 samples the roadmap calls for, `pettingzoo.test.api_test`
passes in both single-deal and full-match episode modes, and suit augmentation is verified to
leave the DD-solved value of a deal exactly unchanged. Four places where this module knowingly
departs from the design spec's literal wording (a smaller action-space dimension, a different
`melds_public` block size, the `"observation"` vs `"obs"` key name, and always computing
`bid_recent` rather than gating it to the auction phase) are each documented at their own site in
[`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) M3 rather than silently
resolved.

Building this also caught a real test-suite pacing bug: several full-8-trick DD-solve tests from
M2 were sitting unmarked in the fast tier, each taking minutes rather than seconds. They're now
correctly tiered as `slow`, with cheap reduced-deal companions covering the same properties — the
fast tier (300 tests) runs in well under a minute.

**M4** (`src/bazarblot/agents/`) adds three baseline agents behind a shared `Agent` protocol
(`act(info, space, legal_mask) -> int`): a uniform-random baseline, a "club player" heuristic
(hand-evaluation bidding + textbook trick-taking play conventions — second hand low, cash aces in
`NT`, lead trump when declaring), and a PIMC agent (sample K worlds respecting known voids,
DD-solve each candidate card, play the argmax). Heuristic beats random by a wide margin — a mean
squashed per-deal margin of ~0.98 out of a ±1 range over 200 deals. Building PIMC needed a new
public entry point in the solver, `solve_from`, since PIMC must evaluate cards from positions
that aren't a fresh trick lead; it's verified against `solve()` and against replaying a snapshot
partway through an already-optimal line.

**"PIMC-20 beats heuristic" is honestly not verified at that scale** — a real, reported
limitation carried over directly from M2's disclosed solver performance. PIMC needs `K x
len(legal_cards)` full DD solves for one decision, and each solve is 1–40+ seconds; PIMC-20's
first play-phase decision alone would need on the order of 160 full solves. What's verified
instead: PIMC taking over as declarer for the final few tricks of many deals (where remaining
hands are small and solves are cheap) scores within a defensible margin of a heuristic declarer
against *identical* defense from the *identical* snapshot. See
[`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) M4 for the full account
of what's verified and what isn't.

**M5** (`src/bazarblot/eval/`) adds the paired/duplicate evaluation harness the whole project
runs on from here — the roadmap's own rule is that no unpaired comparison is trustworthy at any
affordable sample size, given how much Belote's per-deal variance dominates. `duplicate.py`
plays every comparison as a duplicate pair (same shuffle, seats swapped) or duplicate match;
`metrics.py` wraps that in `bootstrap_ci` and two cost tiers — `evaluate_pairs` (cheap, no DD
solving) and `evaluate_dd_oracle_metrics` (expensive, 5+ full solves per deal via the M2.5 batch
solver); `elo.py` runs a round-robin pool on the same paired primitive. The roadmap's own
"done when" bar is met formally, not just observed once: 300 duplicate pairs of random-vs-random
give a paired mean CI that contains 0, and heuristic-vs-random gives a mean margin over 100 with
a CI nowhere near 0 — both in under 8 seconds.

Building the pairing primitive surfaced a real bug worth knowing about: naively letting each side
of a "duplicate" pair redeal its own 4-pass abort (matching how live play behaves) can silently
break the "same shuffle" guarantee the whole method depends on, if one side's bidding aborts a
shuffle the other side's doesn't. Fixed by discarding and resampling the whole pair on any abort
rather than redealing within one side — see
[`docs/03-implementation-roadmap.md`](docs/03-implementation-roadmap.md) M5 for that and a second,
related ordering bug the fix's own test caught. The harness also surfaced a genuine finding about
M4's heuristic bidder (it only competes against an overbidding opponent on inflated, ceiling-capped
estimates, and reliably fails those) — left unfixed here on purpose, since fixing agent quality is
a different task than building the tool that found the issue.

Next: **M6** (throughput, gated on measurement) or **M7** (the first learning run) — M5's harness
is the thing both were waiting on.

The one thing still worth gathering: ~20 real deal lines from the app (cards taken, combinations,
bid, both final scores) as a conformance fixture — the UI's "Copy replay log" button makes this
easy to produce now. It would settle the rounding question empirically and keep re-checking the
scoring code forever after.
