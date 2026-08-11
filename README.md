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
conda env create -f environment.yml && conda activate bazarblot && pip install -e ".[dev]"
```

```bash
pytest -q && ruff check . && mypy
```

## Status

**M0 and M1 complete.** The rules engine (`src/bazarblot/core/`) implements auction, trick play,
combination detection and scoring end to end: `cards.py`, `rules.py`, `declarations.py`,
`auction.py`, `play.py`, `scoring.py`, `deal.py`, `match.py`. 209 tests, 99% coverage of `core/`
(floor is 95%), a golden-hash regression fixture over 1000 seeded deals, and a 1,000,000-deal
random-legal-playout validation with invariants enabled and zero failures.

Two real bugs were caught and fixed during implementation, not just during design:
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

Both corrections are written up in [`docs/01-rules.md`](docs/01-rules.md) §4.1–§4.3 and §7.4, not
just fixed silently.

Two rules questions remain open (auction termination after three passes, and half-up vs half-down
rounding), both behind config flags in `presets/blotstar.yaml` and neither blocking.

Next: **M1.5** — a local UI to watch, play and replay deals. Then M2 (double-dummy solver) and M3
(environment layer) in parallel.

The one thing still worth gathering: ~20 real deal lines from the app (cards taken, combinations,
bid, both final scores) as a conformance fixture — it would settle the rounding question
empirically and keep re-checking the scoring code forever after.
