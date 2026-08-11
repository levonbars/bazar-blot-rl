# Bazar Blot RL Environment — Design Specification

This is the contract an implementer builds against. It is deliberately prescriptive: shapes,
indices, invariants. Read [`01-rules.md`](01-rules.md) first — the rules config drives everything
here.

Package name: `bazarblot`.

---

## 1. Layering

Keep these strictly separated. Most card-game RL codebases rot because the rules engine and the
tensor encoder are the same object.

```
bazarblot/
  core/          pure rules engine. no numpy, no torch, no gym. deterministic, seedable.
    cards.py         card ids, suits, ranks, orderings, point tables
    rules.py         RuleConfig dataclass (mirrors §11 of 01-rules.md), loaded from yaml
    declarations.py  meld detection + comparison
    auction.py       bid legality, auction termination
    play.py          legal-move generation, trick resolution
    deal.py          Deal: full-information state machine (the ground truth)
    scoring.py       raw -> rounded -> match score
    match.py         multi-deal match, dealer rotation, target
  solver/
    dd.py            double-dummy (perfect-information) alpha-beta solver for the play phase
    pimc.py          Perfect-Information Monte Carlo agent (baseline + oracle)
  env/
    obs.py           InfoSet -> tensor encoder (versioned!)
    actions.py       action space, index <-> semantic mappings, legality masks
    aec.py           PettingZoo AEC env (4 agents)
    single.py        Gym-style single-agent wrapper (3 fixed opponents)
    vec.py           batched/vectorized env for throughput
  agents/
    random_agent.py  uniform over legal
    heuristic.py     hand-crafted bidder + player (the "club player" baseline)
    nn/              torch models
  eval/
    duplicate.py     paired-deal evaluation harness
    elo.py
    metrics.py
  cli/
```

**Rule:** `core/` never imports anything from `env/`, `agents/`, or torch/numpy. It should be
importable and testable in isolation, and fast enough to run millions of deals.

---

## 2. Core state model

### 2.1 `DealState` (full information — the simulator's ground truth)

```python
@dataclass
class DealState:
    rules: RuleConfig
    deal_id: int
    dealer: int  # 0..3
    hands: tuple[frozenset[int], ...]  # 4 × set of card ids, mutated as cards are played
    phase: Phase  # AUCTION | DECLARE | PLAY | TERMINAL | ABORTED
    to_act: int

    # auction
    bid_history: list[BidAction]  # full sequence, with seat
    contract: Contract | None  # (level, type, declarer, multiplier)
    consecutive_passes: int

    # combinations — announce/show protocol, rules §6.5
    held: tuple[list[Meld], ...]  # per seat, ground-truth detected melds (PRIVATE)
    announced: tuple[Class | None, ...]  # per seat, the CLASS claimed at hand 1 (PUBLIC):
    # tierce | fifty | hundred | 4x. Need not correspond to
    # anything in `held` — bluffing is legal — so this is a
    # bare enum, never a reference to a real Meld
    answered_top: tuple[Rank | None, ...]  # per seat, sequence top card given when questioned
    answered_trump: tuple[bool | None, ...]  # per seat, the trump bit given when questioned
    answered_rank: tuple[Rank | None, ...]  # per seat, carré rank given when questioned
    # all three PUBLIC and all three possibly false;
    # NOTE: the suit is never disclosed at this stage
    shown: tuple[list[Meld], ...]  # per seat, melds actually revealed (PUBLIC)
    claim_order: list[int]  # announcers ranked; head is the current claimant
    combination_winner: int | None  # team, resolved once a show succeeds or all lapse

    # play
    tricks: list[Trick]  # each: leader, 4 cards in play order, winner, points
    current_trick: Trick | None
    trick_points: list[int]  # per team, running
    belote_holder: int | None

    # match context (read-only during the deal, but part of the observation)
    match_score: tuple[int, int]
    deal_number: int
```

### 2.2 `InfoSet` (what one player can legally know)

Derived from `DealState` by a **single** function `info_set(state, seat) -> InfoSet`. This
function is the security boundary. Write a test that asserts an `InfoSet` never contains a card
id from another seat's unplayed hand — including transitively through any derived field.

`InfoSet` contains:
- own hand, own seat, dealer, match score, deal number
- full bid history (public)
- contract (if set)
- all completed tricks with who played what
- current trick so far
- `announced` (class), `answered_top` / `answered_trump` / `answered_rank`, `shown` for all
  seats — **never `held`**
- everything derivable from the above (voids, remaining cards, running points)

The `held` / `announced` split is the sharpest leakage risk in the whole state model. Four
disclosure levels, and only the last is verified:

| Field | Visibility | Truthful? |
|---|---|---|
| `held` | private forever | ground truth |
| `announced` | public | **unverified** — may be a bluff |
| `answered_top` / `answered_trump` / `answered_rank` | public | **unverified** — may be a bluff |
| `shown` | public | verified |

A bug that lets `held` reach the observation doesn't crash anything — it just quietly hands the
agent a perfect bluff detector and inflates every number in the paper. Test it explicitly:
construct two deals identical in every public field but differing in whether an announcement was
truthful, and assert the encoded observations are bit-identical.

### 2.3 Invariants to assert (cheap, keep them on in tests, off in hot loops)

- `sum(len(h) for h in hands) + 4*len(tricks) + len(current_trick) == 32`
- every card appears exactly once across hands ∪ played
- legal-move set is non-empty whenever `phase == PLAY` and it's someone's turn
- `score_D + score_O == round10(raw_D + raw_O)` (§7.3 of the rules)
- replaying a deal from `(seed, action_sequence)` reproduces byte-identical state

---

## 3. Action space

**One flat discrete space, phase-masked.** Two separate heads is also fine, but a single space
makes the env API uniform and makes it trivial to log/replay trajectories.

Five contract types (`♣ ♦ ♥ ♠ NT`), **no all-trump**. A bid is a **triple**
`(level, type, capot_flag)` — capot is a modifier on a bid, not a separate contract and not merely
an outcome (rules §3.4, §4.4). The level ladder runs `8..80`; 80 is both the theoretical and the
practical maximum, being the largest makeable total (rules §4.2).

| Range | Count | Meaning |
|---|---|---|
| `0` | 1 | `PASS` |
| `1` | 1 | `CONTRA` |
| `2` | 1 | `RECONTRA` |
| `3 .. 3+B-1` | `B = (MAX_BID-MIN_BID+1) × 5 × 2` | `BID(level, type, capot)`, index `3 + 10*(level-MIN_BID) + 2*type + capot` |
| next 32 | 32 | `PLAY(card_id)` |
| next 31 | 31 | combination protocol — only if `rules.declarations_are_actions`; see §3.1 |

With `MIN_BID = 8`, `MAX_BID = 80`: `B = 73 × 5 × 2 = 730`, `ACTION_DIM = 796`.

Define `MIN_BID`, `MAX_BID`, the type list and the capot flag in `RuleConfig` and **derive** the
action space — never hard-code a dimension outside a test that asserts it.

### 3.1 The combination protocol is three decision points, not one

Rules §6.5. Information is released in three widening steps, and the agent may stop at any of them.

| Step | When | Actions | Count |
|---|---|---|---|
| **Announce** | before your hand-1 card | `ANNOUNCE(class)` for `class ∈ {tierce, fifty, hundred, 4x}`, or `SILENT` | 5 |
| **Answer** — sequence | if questioned | `SAY_SEQ(top_card, is_trump)`, `top_card ∈ 8 ranks × is_trump ∈ 2` | 16 |
| **Answer** — carré | if questioned | `SAY_CARRE_RANK(r)` for `r ∈ {J, 9, A, 10, K, Q}` | 6 |
| **Question** | when an opponent has announced and you hold a combination | `ASK`, `DONT_ASK` | 2 |
| **Show** | before your hand-2 card (or the lapse window) | `SHOW`, `WITHHOLD` | 2 |

31 combination actions; `ACTION_DIM = 3 + 730 + 32 + 31 = 796`.

**The answer discloses the top card and a trump bit — never the suit.** "Fifty, to the King, not
trump" localises the holder's four-card length to one of three suits, not one. Encode it that way;
resolving it to a concrete suit internally would hand the agent information the rules withhold.

**`ANNOUNCE`, `SAY_SEQ` and `SAY_CARRE_RANK` are all deliberately unmasked by holdings.** Bluffing
is legal and free, so every class, top card, trump bit and rank stays legal regardless of what the
player holds. This is the only region of the action space where legality is *not* a function of
the private hand — resist the instinct to "fix" it by masking to real combinations, and put an
explicit test on it.

Answering is **compelled but free**: you cannot refuse to answer, yet the content may be a lie.
That is what makes questioning a genuine decision node rather than an automatic state transition,
and it is why `SAY_*` are actions rather than deterministic reveals.

**`SHOW` is masked.** You can only reveal cards you actually hold, and only ones consistent with
what you announced and answered. This is where a bluff finally becomes unrunnable.

Note `4x` hides a 100–200 spread until the rank is given, so it is a strictly coarser signal than
any sequence announcement — the observation must not silently resolve it to a value.

### 3.2 The policy head: relative and factored, not flat

The engine's action space must stay absolute and complete (767). The *policy head* must not mirror
it. Two separate problems, with two separate fixes.

**Problem 1 — absolute levels don't generalize.** 73 independent level logits make "bid 24" and
"bid 25" unrelated parameters, and the upper ladder is vanishingly sparse: bid 80 needs one exact
16-card split plus capot, `P ≈ 3.3 × 10⁻⁹` per deal (rules §4.2). Logits up there receive a
gradient a handful of times in a billion deals.

*Fix: encode the level as a **raise** `Δ` over the standing bid, not as an absolute level.*
"Raise by 2 in hearts" is the same decision at a standing bid of 12 or 42, so the parameters are
shared across the whole ladder. The opening bid uses a virtual standing level of `MIN_BID − 1 = 7`,
so an opening of 25 is `Δ = 18` and needs no special case.

**Problem 2 — the space is multiplicative.** `73 × 5 × 2 = 730` comes from taking the product.

*Fix: factor the head into three conditional sub-heads* — `Δ`, `type`, `capot` — sampled jointly
under the joint legality mask, then mapped back to the flat engine action. The cost becomes
additive: `72 + 5 + 2 = 79` logits (the largest possible raise is `80 − 8 = 72`).

**Together these are lossless.** Every legal bid stays exactly expressible in a single action, at
79 logits instead of 730. Coarsening the raise menu is *not* required to get the compactness, and
should not be done by default.

> **Why not a small fixed menu (`+1, +2, +3, +5, +10`)?** It looks appealing — 5 × 5 × 2 = 50 — but
> raises cannot be chained. After a player bids, the turn passes; if the other three pass, the
> auction closes on that bid and the player never acts again. A hand holding a carré of jacks and a
> Fifty is worth 25 before a card is played (rules §7.5 fixture B), and a policy that can only open
> at 8 will get stuck there. Under this scoring the made-contract payout is `M×bid + collected`, so
> that is a direct ~17-point loss on a hand it should have dominated. The opening bid especially
> must be expressible in one action.
>
> One live caveat: gambler.ru describes a rule where three passes let the **last bidder** raise or
> change their own contract rather than closing the auction. If that applies here it would permit
> self-chaining and weaken this whole argument. Still unresolved — **[OPEN-2c]**, defaulting to
> "auction closes". Keep it behind `rules.three_passes_after_bid` so the head design can be
> revisited without touching the engine.

**Coarsened menus belong in the ablation table, not the default.** A restricted raise menu is a
*bid abstraction*, directly analogous to bet-size abstraction in poker agents, and it is a
legitimate technique — just a lossy one. Because the engine stays unabstracted, the abstraction
lives purely in the agent layer and its cost is measurable: run the abstracted agent against the
unabstracted one, and report the fraction of double-dummy-oracle bids the menu cannot express.
That comparison is a genuine paper result; adopting the menu silently is not.

**Never lower `MAX_BID` in training** to shrink the space. A truncated ladder makes legal auction
states unrepresentable and the env will throw on a deal that was legal.

Diagnostics for the first run: the empirical raise-size and absolute-level histograms, and the
fraction of the flat space ever legal. "The action space is 767-wide but its effective support is
~40 levels and a handful of raise sizes" is an observation about the game worth reporting.

Provide `legal_mask(info_set) -> np.ndarray[bool, ACTION_DIM]` in `env/actions.py`, computed from
`core` legality functions. **The mask must be computed from the `InfoSet`, not the `DealState`** —
otherwise you can leak information through the mask (a classic bug: a mask computed from full
state can differ from one computed from the info set, and the agent learns to read it).

Typical legal counts: auction 1–30, play 1–8. The mask is extremely sparse; log
`mean(legal_count)` per phase as a training diagnostic.

---

## 4. Observation encoding

`env/obs.py` exposes `encode(info_set) -> dict[str, np.ndarray]` and a version string
`OBS_VERSION = "v1"`. **Bump the version on any change and refuse to load checkpoints trained on
a different version.**

### 4.1 Seat-relative canonicalization (do this)

Encode everything relative to the acting player: `rel = (other_seat - my_seat) % 4`, giving
`0 = me, 1 = left opponent, 2 = partner, 3 = right opponent`. A single network then serves all
four seats and you get a free 4× in effective sample size.

### 4.2 Suit canonicalization + augmentation (do this too)

Suits are interchangeable except for their role in the contract.
- Under `TRUMP_x`: relabel so trump = suit 0; the other 3 suits are freely permutable → **6×**
  data augmentation, and canonical ordering (e.g. by length in hand) is also an option.
- Under `NT` ("boy"): no suit has a special role, so all 4 suits are permutable → **24×**.

Apply as a random permutation at sample time in training, and average over permutations at
inference for a cheap ensemble. This is one of the highest value-per-line changes in the whole
project; do it in M3, not as an afterthought.

### 4.3 Feature blocks (`v1`)

| Block | Shape | Description |
|---|---|---|
| `hand` | 32 | binary, own cards still held |
| `card_state` | 32 × 6 = 192 | per card one-hot: `{in my hand, played by rel-0, rel-1, rel-2, rel-3, unseen}` |
| `current_trick` | 32 × 4 = 128 | per card, which rel-seat has played it **into the current trick** (zero if not) |
| `trick_leader` | 4 | rel seat leading current trick |
| `trick_index` | 8 | one-hot 0..7 |
| `contract_type` | 6 | one-hot `{none, C, D, H, S, NT}` |
| `contract_level` | 25 + 1 | one-hot for levels 8..32 (clipped), plus scalar `level/32` — do **not** one-hot all 73 rungs |
| `capot_flag` | 1 | contract was bid as capot; the shutout is then mandatory |
| `declarer_rel` | 4 | which rel seat is the attacking bidder |
| `doubling` | 3 | one-hot `{none, contra, recontra}` |
| `bid_payout` | 1 | `M × level / 32` — the *realised* stake, since `M` differs for trump vs `NT` (rules §7.2) |
| `bid_summary` | 4 × 10 = 40 | per rel seat: `has_passed`(1), `n_bids`(1), `last_level` scalar(1), `last_capot`(1), `last_type` one-hot(6) |
| `headroom` | 2 | `(MAX_BID − standing_level)/72` and the standing level itself — the relative bid head needs to know how much ladder is left, since large `Δ` is illegal near the top |
| `bid_recent` | 8 × 16 = 128 | last 8 auction actions, each as `(kind one-hot 4, type one-hot 6, level scalar, capot, seat rel 4)` — a compact tuple, **not** a one-hot over the flat action space **(auction phase only)** |
| `own_combinations` | 12 | my detected combinations: best sequence value, best carré value, Blot-Reblot, and how much they contribute to a bid — the single most important bidding feature, since combinations count toward fulfilment |
| `voids` | 3 × 4 = 12 | inferred: rel seat `r` failed to follow suit `s` at some point |
| `cards_left` | 4 | per rel seat, `n/8` |
| `suit_unseen` | 4 | count of each suit not yet visible to me, `/8` |
| `melds_public` | 4 × 25 = 100 | per rel seat, one channel per disclosure step: announced class one-hot (5: none/tierce/fifty/hundred/4x), answered top card one-hot (9: none + 8), answered trump bit (2: unknown/yes-no), answered carré rank one-hot (7: none + 6), `was_questioned`(1), `has_shown`(1), `announced_but_withheld`(1), belote(1), rebelote(1). Keep the disclosure levels as **separate** channels — the gaps between them are the bluff signal, and collapsing them to a resolved value destroys the sub-game. There is deliberately **no suit channel** before the show |
| `running_points` | 4 | my team raw pts, opp raw pts, both `/162`; tricks won each `/8` |
| `contract_progress` | 5 | `raw_A/(10·level)`, `points_still_available/162`, `sign(need_met)`, `capot_still_live` (attackers have taken every trick so far), `capot_already_broken` (a capot *bid* is now unmakeable) |
| `match_state` | 6 | my match score, opp match score (both `/target`), `points_to_target` each, deal number, dealer rel |

**Total (play net, dropping `bid_recent`): ≈ 480 floats.**
**Total (auction net): ≈ 610 floats**, or keep the auction history as a length-`T` token sequence
for a small transformer instead of a flat block (recommended if you go the transformer route).

Note the level is encoded as a **clipped one-hot plus a scalar**, not a 73-way one-hot. Levels
above ~32 are rare enough that spending 73 input dims on them is waste; the scalar carries the
magnitude and the one-hot carries the fine structure where the density actually is.

The two capot features carry a lot of weight here. Capot sets the card portion to 252 raw = 25
scaled (rules §7.3), well over a normal deal's whole 16, and it can be either bid or stumbled
into. `capot_already_broken` is the sharper of the two: the instant the defenders take a trick
against a bid capot, the contract is dead no matter how many points the attackers go on to
collect, and the correct play switches from maximizing tricks to damage limitation. An agent
without that feature has to infer it from the trick history every ply.

### 4.4 Alternative: sequence encoding

Everything above is a hand-engineered flattening. A cleaner and probably stronger design for the
paper is: represent the deal as a **token sequence** (one token per public event: bid, card
played, announcement, show/withhold) + a private hand embedding, and run a small causal
transformer. It removes the `bid_recent` hack, handles variable-length auctions natively, handles
the contingent declaration windows without special-casing, and makes both phases share one trunk.

**Recommendation:** ship `v1` flat features first (they are debuggable and fast), and treat the
transformer as the M6 upgrade with the flat-feature model as the ablation baseline. "Do features
matter, or does the sequence model learn them?" is a real result worth a table in the paper.

---

## 5. Environment APIs

### 5.1 `PettingZooAEC` (primary)

Agent ids `"player_0".."player_3"`. Standard AEC loop: `reset(seed)`, `agent_selection`,
`observe(agent) -> {"obs": ..., "action_mask": ...}`, `step(action)`, `last()`.

- One **episode = one deal** by default (`episode_unit="deal"`), or one full match
  (`episode_unit="match"`). **Both are first-class** and the owner wants single-deal training
  available. Deal-level episodes with match score in the observation is the recommended default:
  shorter credit-assignment chains, and the match context is still visible.
- Caveat to state in the paper: deal-level episodes **cannot** learn endgame bidding ("we need 12
  to win, bid accordingly"), because the terminal reward never depends on crossing the target. At
  a 101 target that behaviour is a real part of the game. Train on deals for the play policy,
  fine-tune or evaluate on matches, and report both — don't let a deal-trained number stand in for
  match strength.
- Illegal action → raise in `strict` mode (dev), or mask-and-resample in `permissive` mode
  (never silently no-op; you'll spend a week debugging it).

### 5.2 `SingleAgentEnv` (convenience)

Gym API. One learning seat, three fixed policies (from a pool / league). Rotates the learner's
seat each episode. Used for evaluation and for quick baselines, not for self-play training.

### 5.3 `VecEnv` (throughput)

Batched stepping of `N` independent deals. Target: **≥ 50k deals/s** on one CPU core for the pure
engine, **≥ 5k deals/s** including encoding. Get a Python reference implementation correct first,
measure, and only then decide whether to port `core/` to Rust (via `maturin`/PyO3) or C++. Expect
the naive Python engine to be ~2–5k deals/s — enough for M1–M4, likely not enough for the final
training runs.

---

## 6. Reward

Let `Δ = score_my_team − score_opponent_team` at the end of a deal, in match points as computed by
rules §7.2.

| Scheme | Definition | Use |
|---|---|---|
| `deal_margin` (default) | `squash(Δ)`, see below | main training signal |
| `deal_win` | `sign(Δ)` | ablation; loses magnitude info, don't use as primary |
| `match_win` | `±1` at match end only | fine-tuning, and the honest headline metric |

**The reward scale is not what classical Belote would give you, and it is heavy-tailed.** A deal
does not distribute a fixed 16 points: a made contract pays the attackers `M × bid` *on top of*
their collected points, combinations count toward the bid so the ladder runs far above 16, and a
failed contract pays the defenders `M × bid + 16 + bonuses`.

| Outcome | Attackers | Defenders |
|---|---|---|
| trump made, bid 8 | `8 + collected` ≈ 16–24 | ≈ 0–8 |
| trump made, bid 16 | `16 + 16 = 32` | 0 |
| trump failed, bid 8 | 0 | `8 + 16 = 24` |
| trump made, bid 26 on a jacks-carré hand (rules §7.5 fixture B) | `26 + 30 = 56` | 11 |
| `NT` failed, bid 16, recontra | 0 | `5×16 + 16 = 96` |
| `NT` failed, bid 40, recontra | 0 | `5×40 + 16 = 216` |

The tail is genuinely long — a recontra'd no-trump failure at a high level is two orders of
magnitude above a quiet made contract. **Do not use a plain linear `Δ / NORM`.** A single such
deal will dominate a batch's gradient, and these deals are exactly the rare ones where the
policy's estimate is worst.

Recommended: `squash(Δ) = tanh(Δ / 24)` (or an asinh, which preserves more resolution in the
tail), with `24` set from the empirical median absolute margin rather than the maximum. Log the
raw-margin histogram in the first run, fix the constant once, and freeze it — changing it later
silently rescales the value function and invalidates cross-run comparisons.

Report *unsquashed* margins in evaluation. The squash is a training device, not a metric.

**Zero-sum: only conditionally.** The two teams' deal scores do **not** sum to a constant under
this scoring — both teams gain on a made contract (attackers `M×bid + collected`, defenders their
own `collected`, confirmed by the owner). Use the *margin* `Δ` as the reward, which is
antisymmetric by construction, and assert `reward_A == -reward_B`. Do **not** assert that raw deal
scores sum to anything.

**Do not shape.** No per-trick rewards, no "you made your contract" bonus. The deal is 32 plies —
short enough that sparse terminal reward works, and shaping is exactly what will make the bidding
policy learn something subtly wrong. If credit assignment stalls, use auxiliary *prediction* heads
instead (§7.2), not reward shaping.

**Variance reduction is mandatory.** Deal variance dwarfs policy differences in Belote. Use
duplicate/paired sampling (§9) both in evaluation *and*, if feasible, in training (same deal
played by both seat-assignments in the same batch, so deal luck cancels within the batch — this is
the card-game analogue of an antithetic-variates baseline, and it is cheap).

---

## 7. Model and algorithm notes

### 7.1 Why not literally AlphaZero

AlphaZero's PUCT search assumes perfect information: a node is a state, and the search tree is the
game tree. Here a node is an *information set*, and naive MCTS over it is unsound (strategy fusion
and non-locality). Say this explicitly in the paper — it's the framing.

Viable families, in the order I'd build them:

1. **Model-free self-play baseline.** PPO or IMPALA with action masking, or a DouZero-style deep
   Monte-Carlo (regression onto returns per legal action). Cheap, robust, gives you an honest
   floor. *Ship this first.*
2. **PIMC / determinized search + learned evaluation.** Sample `K` worlds consistent with the
   info set, solve or roll out each, aggregate. Known to be strategy-fusion-biased but very strong
   in practice in trick-taking games (it is what most commercial Bridge/Skat bots do). Excellent
   *baseline* and excellent *bid evaluator*.
3. **Information-set search with a learned belief** (Player-of-Games / Suphx flavour): re-solve
   subgames from a public belief state, with a network supplying leaf values and a belief model
   supplying the world distribution.
4. **Regret-based**: Deep CFR / ReBeL on the play subgame given a fixed contract. The play phase
   after the auction is a small enough public-state game that ReBeL-style public-belief-state
   value functions are plausible. This is the most novel and the most likely to eat six months.

The honest research contribution is most likely **(a)** the two-phase coupling (how do you train a
bidder whose reward depends on your own play strength, without it collapsing?), and **(b)** the
cooperative-adversarial mix (partner is a separate agent with different private info — no cheap
centralized-critic shortcut unless you're careful about what the critic sees at *execution* time).

### 7.2 Auxiliary heads (cheap, high value)

Train alongside the policy/value:
- **Belief head:** for each of the 24 unseen cards, `P(held by rel-1 / rel-2 / rel-3)`. Supervised
  from the ground-truth deal, free labels. Strong regularizer and directly usable to sample worlds
  for PIMC/ISMCTS.
- **Bluff-detection head** (only with `declarations_are_actions`): given an opponent's
  announcement, `P(they actually hold it)`. Free labels from the ground-truth deal. Without this
  the agent has to learn the entire announce/show sub-game from the terminal reward alone, which
  is a very thin signal for a decision that occurs once per deal.
- **Double-dummy head:** predict the DD-optimal number of tricks / points for this info set
  (labels from `solver/dd.py`). Turns the exact solver into a distillation target.
- **Contract-outcome head:** `P(contract made)` and expected `raw_D`. This is *the* bidding signal.

### 7.3 The two-phase training problem (state it as the paper's core question)

The bidder's value depends on the player's skill. Options to compare:
- **Joint end-to-end self-play** — one policy, both phases, terminal reward only.
- **Two-stage / iterated:** freeze play policy → train bidder against it → freeze bidder → retrain
  player → repeat. Cleaner credit assignment, risk of oscillation.
- **Bid-by-evaluation:** the bidder does not learn a policy at all; it evaluates each candidate
  contract with the *play* network's value head (or with PIMC/DD rollouts) and bids argmax. This
  is the strongest simple baseline and mirrors how strong human players actually bid.

A results table comparing these three is a publishable core of the paper.

### 7.4 Centralized training, decentralized execution

The critic may see the full deal (all four hands) during training; the actor must not. Enforce
this with types: `Critic.forward(full_state)`, `Actor.forward(info_set_obs)`, and never let an
`InfoSet` and a `DealState` be the same object. Note in the paper whether the critic is
centralized — reviewers will ask.

---

## 8. The double-dummy solver (build it early — M2)

Perfect-information solve of the 32-ply play phase given all four hands and a contract.

- Alpha-beta over "points taken by the declaring team", with:
  - transposition table keyed by `(cards remaining bitmask ×4, to_act, trick-so-far, points)` —
    use the standard trick of keying on the *equivalence-reduced* position (adjacent cards of
    equal rank in the same suit that are all still out are interchangeable)
  - move ordering (high cards / winning ruffs first), and
  - "quick tricks" cutoffs.
- Expect microseconds to low milliseconds per deal after equivalence reduction.

**Why it earns its place:**
1. It is the strongest possible correctness test of `core/play.py` and `core/scoring.py`
   (cross-check with an independent implementation, e.g. by adapting NikolayIT's Bulgarian engine).
2. It gives **bid-quality ground truth**: for a dealt hand, enumerate all contracts, DD-solve each
   over sampled partner/opponent distributions, and you get an "oracle bid". Reporting the agent's
   bidding accuracy against a DD oracle is a far more informative metric than winrate alone.
3. It makes PIMC possible, which is your strongest non-learned baseline.
4. It supplies distillation labels for §7.2.

---

## 9. Evaluation harness

Belote's per-deal variance is brutal: two identical policies can differ by many points per 100
deals purely from deals. Design for this from day one.

- **Duplicate (paired) evaluation.** Play every deal twice: once with policy X at seats {0,2} and
  Y at {1,3}, once with the assignment swapped, using the *same shuffle*. Report the paired
  difference. This kills most of the deal variance and is standard in duplicate bridge.
  Implement in `eval/duplicate.py`; make it the only evaluation entry point.
- **Metrics:**
  - IMP-like / raw points per deal (paired), with bootstrap CI
  - match winrate at the configured target
  - contract make rate; over-bid rate; under-bid ("left points on the table") rate
  - bid agreement with the DD oracle (exact / within-1-level)
  - trick-play efficiency: points lost vs. DD-optimal play *given the actual deal*
    (i.e. how much worse than a cheating player), split by contract type
  - declaration handling, if `declarations_are_actions`
- **Opponents:** random-legal, heuristic, PIMC-`K` for several `K`, prior checkpoints (league),
  and if you can get them, logs from a real platform.
- **Elo** over a round-robin pool, computed on paired deals.

---

## 10. Reproducibility

- Every deal generated from `(match_seed, deal_number)` via a counter-based RNG (Philox / SplitMix)
  so any deal is reconstructible without storing it.
- Trajectory log format: `(rules_hash, obs_version, match_seed, deal_number, dealer, action_seq)`.
  That plus the engine reconstructs everything; do not log observations.
- Stamp `rules_hash` + `OBS_VERSION` + git sha into every checkpoint; refuse mismatched loads.

---

## 11. Open design questions to settle with the user

1. **Two `[OPEN]` rules gaps remain** (`01-rules.md` §12): OPEN-2c (auction termination) and
   OPEN-10 (rounding). Neither blocks M1. Everything else is resolved.
2. Episode unit: deal (default) or match? With a 101-point target and deals worth up to ~32,
   a match is only 4–6 deals, so `episode_unit="match"` is more affordable here than it would be
   in a game with a 1000-point target — worth reconsidering rather than defaulting.
3. Are declarations actions (concealment allowed) in the headline experiments, or auto-announced?
4. Is there any source of human game logs (a platform API, personal logs)? If yes, supervised
   pretraining changes the whole plan for the better and should become M5.
5. Target compute budget. This decides Python-vs-Rust for `core/` and whether search-based methods
   are on the table at all.
