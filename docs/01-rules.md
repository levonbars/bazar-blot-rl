# Bazar Blot (Armenian Bazaar Belote) — Formal Rules Specification

**Canonical source: [Blot Star](https://blotstar.com/bazar-blot-rules/)** — the ruleset the project
targets, chosen by the project owner. Where Blot Star is explicit, it wins over every other
source, including classical French Belote and other Armenian platforms.

Three tiers of claim are marked throughout:

- **[BS]** — stated directly on the Blot Star rules page. Treat as ground truth.
- **[OWNER]** — from the project owner, who plays the game. **Overrides [BS] on conflict.**
- **[DERIVED]** — not stated, but forced by arithmetic consistency with a **[BS]** claim. Safe,
  but the derivation is shown so it can be checked.
- **[OPEN]** — still unresolved. Only **two** remain (§12), neither blocking; both sit behind
  config flags in `presets/blotstar.yaml`. Defaults given are documented fallbacks, not evidence.

---

## 1. Overview

Bazar Blot is a 32-card, four-player, 2-vs-2 partnership trick-taking game of the Belote family,
played in Armenia and the Armenian diaspora. It replaces classical French Belote's "turn one card
up, accept-or-pass" trump determination with a full **auction** (*bazar* = market / bargaining)
held before any card is played. **[BS]**

Two structurally distinct phases per deal:

| Phase | Nature | Length |
|---|---|---|
| **Bazar** (auction) | sequential bidding under private information | variable |
| **Trick play** | 8 tricks × 4 cards, under follow/trump obligations | exactly 32 actions |

The auction fixes *what game is played* (trump suit or no-trump), *which team is committed to a
numeric target*, and *the stake multiplier*. The play phase decides whether the commitment is met.

### 1.1 Terminology (Blot Star's words, used throughout) **[BS]**

| Term | Meaning |
|---|---|
| **Hand** | one trick (4 cards). A round consists of 8 hands. |
| **Round** | one deal. Called "deal" in the rest of these docs. |
| **Boy** | the no-trump contract ("play without a trump") |
| **Blot-Reblot** | King + Queen of the trump suit held together |
| **Cut with a trump** | playing a trump when void in the led suit |
| **Contra** | defending team challenges that the attackers will not reach their announced score |
| **Recontra** | attacking team reaffirms after a contra |
| **Capot** | one team fails to take a single trick all round. Worth 90. |
| **Last Hand / "says"** | winning the 8th trick, worth an extra 10 |

Teams are **Blue** and **Red** on Blot Star. **[BS]**

---

## 2. Players, seating, deck

- 4 players, 2 teams of 2, partners opposite. **[BS]**
- Deck: 32 cards — `7, 8, 9, 10, J, Q, K, A` in `♣ ♦ ♥ ♠`. **[BS]**
- Each player is dealt **8 cards**, then bidding begins. **[BS]**
- Card id encoding (canonical for the implementation):
  `card_id = 8 * suit + rank_index`, `suit ∈ {0:♣, 1:♦, 2:♥, 3:♠}`,
  `rank_index ∈ {0:7, 1:8, 2:9, 3:10, 4:J, 5:Q, 6:K, 7:A}`.
- Dealer rotates one seat **clockwise** each deal; the player to the **dealer's left** opens the
  auction **and** leads to trick 1. **[OWNER]**

---

## 3. Contracts

**There are exactly two contract families: a named trump suit, or "boy" (no trump). [BS]**

> There is **no all-trump contract** in this ruleset. Earlier drafts of this document included one
> from other platforms' rules; it has been removed. Do not reintroduce it.

| Type | Code | Deal card points |
|---|---|---|
| Trump suit | `TRUMP_C / TRUMP_D / TRUMP_H / TRUMP_S` | 152 + 10 last hand = **162** |
| No trump ("boy") | `NT` | 152 + 10 last hand = **162** |

Both families total **162 raw = 16 scaled**, which is exactly why the "+16 points" in the scoring
formulas (§7) is the same number for both. **[DERIVED, and it is the key consistency check]**

### 3.1 Card ranking (trick-taking strength) **[BS — read off the rules page card images]**

- **Trump suit** (in a trump contract): `J > 9 > A > 10 > K > Q > 8 > 7`
- **Plain suit** (non-trump suit in a trump contract) and **all suits under `NT`**:
  `A > 10 > K > Q > J > 9 > 8 > 7`

### 3.2 Card point values

| Card | Trump suit **[BS]** | Plain suit **[OWNER]** | Under `NT` **[BS]** |
|---|---|---|---|
| J | **20** | 2 | 2 |
| 9 | **14** | 0 | 0 |
| A | 11 | 11 | **19** |
| 10 | 10 | 10 | 10 |
| K | 4 | 4 | 4 |
| Q | 3 | 3 | 3 |
| 8 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 |
| **suit subtotal** | 62 | 30 | 38 |

**Confirmed by the owner**, and the derivation is kept because it is a useful invariant to assert.
Blot Star publishes only two tables ("With Trump" =
20/14/11/10/4/3/0/0 and "Without Trump" = 19/10/4/3/2/0/0/0) and does not state what a *side suit*
is worth inside a trump contract. It is forced: a trump deal must total 162 raw for the "+16
points" in §7 to be the same figure as in a no-trump deal, and
`62 (trump suit) + 3 × 30 (side suits) + 10 (last hand) = 162`. Using the 19-point ace in side
suits would give `62 + 3×38 + 10 = 186`, contradicting **[BS]**. So side suits use the classical
Belote values with **A = 11**.

Assert in tests:
- trump contract: `62 + 3×30 = 152`, `+10 = 162`
- no-trump contract: `4×38 = 152`, `+10 = 162`

### 3.3 Last hand ("says") **[BS]**

Winning the 8th trick adds **10** raw points, in **both** trump and no-trump contracts
**[OWNER]**. Included in the 162 above.

### 3.4 Capot **[OWNER]**

Capot = one team takes **all 8 tricks** (the other fails to take a single one).

Blot Star's page describes capot only as an outcome "worth 90". The project owner corrected this
on both counts, and the correction governs:

1. **Capot does not *add* 90 — it *sets* the attackers' card-point portion to 252.**
   Numerically identical to `162 + 90` for the card points alone, but the framing matters: on
   capot the card portion is a flat **252 raw = 25 scaled**, and combination bonuses are added on
   top of that, not folded into it.
2. **Capot is also a bid modifier, not only an outcome.** A bid carries an optional capot flag, so
   `26` and `26 Capot` are two different bids. See §4.

So a bid is a triple: **`(level, contract_type, capot_flag)`**.

---

## 4. The auction (*bazar*)

Blot Star names the mechanics — bid, pass, contra, recontra — but does not publish the ladder.
Everything below comes from the project owner **[OWNER]**. One gap remains, **[OPEN-2c]** in §4.2.

### 4.1 Actions

| Action | Notes |
|---|---|
| `PASS` | **not binding** — a player who passed may bid again on a later turn **[OWNER]** |
| `BID(level, type, capot)` | `type ∈ {♣, ♦, ♥, ♠, NT}`; `capot ∈ {false, true}`; must exceed the standing bid |
| `CONTRA` | available to the **defending** team against a standing contract; opens a two-ply reply window (§4.3) |
| `RECONTRA` | available to the **declarer** (the specific seat who placed the standing bid), in direct reply to a contra; closes the auction |

**Correction (superseding an earlier draft of this doc).** An earlier version of this table said
both `CONTRA` and `RECONTRA` "end the auction" — self-contradictory, since it also describes
`RECONTRA` as coming *after* a contra within the same auction. If contra closed the auction
outright, recontra could never fire. §4.3 gives the corrected sequencing: contra does **not**
close the auction by itself; it hands a single, forced reply to the declarer.

### 4.2 The ladder **[OWNER]**

- Minimum opening bid **8**, minimum raise **+1** **[OWNER]**, on the **scaled** (÷10) axis — a
  bid of 12 commits to 120 raw points **[OWNER]**.
- **The ladder is not capped at 16.** Bids above the deal's 162 card points are legal and normal,
  because **combination bonuses count toward fulfilment** (§7.2) and because capot raises the card
  portion to 252. A team holding a carré of jacks (200) and a Fifty (50) already has 250 raw = 25
  scaled before a single trick is played, so opening around 25 is sound for them and the opponents
  can answer 26.
- **Maximum bid is 80 [OWNER].** The ceiling is a team holding all four scoring carrés split
  across the partners — one player with jacks + nines, the other with tens + aces, exactly 16
  cards — and taking capot:

  | Component | Raw | Scaled |
  |---|---|---|
  | Capot (card portion set to 252) | 252 | 25 |
  | Carré of jacks | 200 | 20 |
  | Carré of nines | 140 | 14 |
  | Carré of aces | 110 | 11 |
  | Carré of tens | 100 | 10 |
  | **Total** | **802** | **80** |

  Both routes agree here (`round10(802) = 80` and the scaled components sum to 80), but that is a
  coincidence of this hand — **the engine must sum raw and scale once** (§7.4), never scale each
  component and add.

  Bidding above 80 is unmakeable by construction, so `MAX_BID = 80` is both the theoretical and
  the practical cap. It is also astronomically rare: it needs one specific 16-card split, so
  `P ≈ 2 / C(32,16) ≈ 3.3 × 10⁻⁹` per deal — about three occurrences in a billion self-play deals.
  Levels above roughly 40 will essentially never be sampled; see the environment spec §3.1 for how
  to size the policy head accordingly.
- **Seniority is purely numeric. [OWNER]** Suits are not ranked against each other, `NT` does
  **not** outrank a suit at equal level (you cannot say "12 boy" over "12 hearts"), and capot does
  **not** outrank a plain bid at equal level either (§4.4). To outbid anything you must raise the
  number.
- 4 passes with no bid → redeal.
- 3 passes after a bid → **[OPEN-2c], deferred by the owner.** Either the auction closes on the
  standing contract, or the last bidder gets a final option to raise or change it (gambler.ru
  describes the latter). Default: **closes**. This is the last structural unknown in the auction;
  if the second reading is right, a player can climb the ladder without opposition, which changes
  both the auction game tree and how much single-action bid expressiveness matters.

### 4.3 Contra and recontra: a two-ply reply **[ENGINEERING — see correction above]**

This section did not exist in earlier drafts; it replaces the self-contradictory "both end the
auction" claim in §4.1 with a sequencing that actually makes recontra reachable. It was derived
during M1 implementation from standard Belote-family convention, not dictated by Blot Star or the
owner — flag it for confirmation next time the owner plays a hand where it comes up.

`CONTRA` does **not** close the auction by itself. It hands a single, forced reply to the
**declarer** — the specific seat who placed the standing bid, not just "their team" — with exactly
two legal responses:

- **`RECONTRA`**: doubling becomes `recontra`. The auction closes.
- **`PASS`** (accepting the contra): doubling stays at `contra`. The auction closes.

No other action is legal in this window — not a new bid, not a pass from anyone else. The
declarer's partner has no say (real-table convention: only the player who named the contract may
redouble it). Once the reply lands, the auction is over; there is no further round of doubling.

`CONTRA` said out of turn is a real-table behaviour; **the environment takes it only on the
player's own turn**. Record this simplification in the paper.

### 4.4 Capot is sticky upward **[OWNER]**

Once anyone bids capot, **every subsequent bid must also be capot.**

> Worked through, in the owner's own example. You hold jacks + a Fifty and can support 25. An
> opponent may answer either `26` or `26 Capot`.
> - If they said plain `26`, going to 27 is an ordinary raise: bid `27`.
> - If they said `26 Capot`, you cannot bid plain `27` — you must bid `27 Capot`.

**Capot does not outrank a plain bid at the same level. [OWNER]** You cannot answer a standing
plain `26` with `26 Capot` — you must go to `27 Capot`. Seniority is purely numeric; the capot
flag is a *commitment* attached to a bid, never a tie-breaker.

So the flag has exactly two effects on legality:
1. it becomes **mandatory** on every bid above a standing capot bid, and
2. it can only ever be added while also raising the number.

### 4.5 Attacking / defending team

The team making the final bid is the **attacking team** (contractors); the other is **defending**.
Both members of the attacking team are bound by the contract; there is no dummy — the bidder's
partner plays their own cards normally.

---

## 5. Trick play

Blot Star's rules page states only the *concept* ("cut with a trump: playing a trump when the
required suit is not available"). The full obligation set below was confirmed by the project
owner **[OWNER]** and matches standard Belote.

Let `L` = led suit, `T` = trump suit.

1. **Follow suit.** Holding a card of `L`, you must play one of `L`.
2. **Beat within trump.** If `L == T` and you hold `L`, you must play a trump higher (in trump
   order) than the highest trump on the trick, if you hold one. **[OWNER]**
3. **Ruff obligation** (trump contracts, when void in `L`): if an **opponent** is currently
   winning the trick and you hold trump, you must play trump. **[OWNER]**
4. **Over-trump obligation.** When trump has already been played to the trick and you are playing
   a trump, you must play a higher one if you hold it; otherwise you may under-trump
   ("pisser"). **[OWNER]**
5. **Partner-winning exemption.** If your **partner** is currently winning the trick, you are
   **not** obliged to ruff — free discard. **[OWNER: confirmed, the stricter "must ruff anyway"
   variant found in terz.am does NOT apply here.]**
6. **`NT` ("boy")**: rule 1 only. No trumping exists; free discard when void.

**Trick winner:** highest trump if any trump was played, else the highest card of `L` in the
applicable order (plain order in a trump contract's side suits and everywhere under `NT`).
Winner leads the next trick; winner of trick 8 takes the 10-point *says*.

---

## 6. Bonus combinations (declarations)

### 6.1 Sequences **[BS]**

Consecutive cards **in natural rank order** `7 8 9 10 J Q K A`, same suit. **Not** trump order —
Blot Star's own illustration of a "Hundred" is `7-8-9-10-J`, which confirms natural order and is a
classic implementation bug worth a dedicated test.

| Name | Cards | Value |
|---|---|---|
| **Tierce** | 3 sequential, same suit | **20** |
| **Fifty (50)** | 4 sequential, same suit | **50** |
| **Hundred (100)** | 5 sequential, same suit | **100** |

Runs of **6, 7 or 8 cards score 100** — the same as a five-run, with no higher rung and no
splitting into smaller combinations. **[OWNER]** Each card belongs to at most one declared
sequence; take the maximum-value decomposition.

### 6.2 Blot-Reblot **[BS]**

**King + Queen of the trump suit** in one hand = **20**.

Only exists in a trump contract (there is no trump suit under `NT`). **[OWNER]**

**Blot-Reblot takes priority regardless of other combinations [BS]** — i.e. it is scored
unconditionally and is not suppressed by §6.4's comparison.

### 6.3 Four-of-a-kind ("Card Points (4X)") **[BS]**

Values differ by contract type — note this carefully, they are not the same table:

| Four of | In a **trump** contract | Under **`NT`** |
|---|---|---|
| Jacks | **200** | 100 |
| Nines | **140** | **0** |
| Aces | **110** | **190** |
| Tens | 100 | 100 |
| Kings | 100 | 100 |
| Queens | 100 | 100 |
| Eights | 0 | 0 |
| Sevens | 0 | 0 |

Two Blot-Star-specific values worth flagging, because they differ from classical Belote and from
other Armenian platforms:
- **four nines = 140** in a trump contract (classical Belote: 150), and **0** under `NT`
- **four aces = 110** in a trump contract (classical Belote: 100), **190** under `NT`

The pattern is coherent: the 4X value tracks the card's point value in that contract type
(J and 9 are worthless under `NT`, so their carrés are worth 100 / nothing; the ace is the big
card under `NT`, so its carré is 190).

### 6.4 Priority between combinations **[BS]**

Blot Star's page gives four loose rules; the owner's precedence ladder is exact and supersedes
them. Implement **this**, in order:

**Step 1 — carré beats sequence, always. [OWNER]** Any four-of-a-kind outranks any sequence,
regardless of value. A carré of queens (100) beats a Hundred (100), and beats it by category, not
by arithmetic.

**Step 2 — carré vs carré:** by value, `J 200 > 9 140 > A 110 > 10/K/Q 100`. Two 100-carrés
(tens vs kings vs queens) tie → elder hand.

**Step 3 — sequence vs sequence:**
1. **Longer wins** — Hundred > Fifty > Tierce.
2. Equal length → **higher top card wins**.
3. Equal top card → **trump wins**.
4. Still equal → **elder hand**.

> **Top card outranks trump. [OWNER]** The owner's worked example:
> **`A-K-Q` non-trump  >  `J-10-9` trump  >  `J-10-9` non-trump.**
> All three are Tierces. The Ace-topped one wins despite not being trump (step 3.2); trump only
> separates the two J-topped ones (step 3.3).

**Step 4 — Blot-Reblot sits outside all of this** (§6.2): flat, unconditional +20 to its holder,
never compared, never suppressed.

Note how cleanly this explains the questioning protocol (§6.5). After the announced class fixes
step 3.1, the only things still needed are the **top card** (3.2) and the **trump bit** (3.3) —
which is exactly what a question discloses, in exactly that order. The suit is never asked because
the ranking never needs it. The disclosure is precisely sufficient and not one bit more.

**Consequence of losing the comparison [OWNER]:** the team holding the winning combination scores
**all** of its combinations; the losing team scores **none** of theirs.

**Blot-Reblot is exempt from the whole comparison. [OWNER]** The owner's words: it "does not
clash with any others". So it survives even when the holder's team loses the comparison — score it
as a flat, unconditional +20 to its holder and resolve everything else independently. This matches
Blot Star's rule 4, and it implements cleanly as a single special case evaluated *before* the
comparison runs.

> One inference to confirm: "does not clash" is read here as **symmetric** — Blot-Reblot is not
> suppressed by a bigger combination, and it also does not *win* the comparison on behalf of its
> team's other combinations. The owner confirmed the first direction explicitly; the second is
> inferred from the same phrase. If it turns out Blot-Reblot's 20 also competes in the ranking,
> only `blot_reblot_exempt_from_comparison` changes.

**Ties [OWNER]:** two equal combinations with neither in trump (both teams hold a Tierce to the
King, say) → **elder hand wins**, i.e. the tied holder earliest in turn order starting from the
player who leads trick 1, and that player's team scores all its combinations. Note the leader is
the dealer's left (§2), so "elder" is measured from there, not from the declarer.

### 6.5 The announce / show protocol **[OWNER]**

Claiming a combination is a **two-stage** act, and the two stages are separate decisions. This is
not a binary "declare or conceal".

**Stage 1 — announce a *class*, at hand 1. [OWNER]** Before playing your card to trick 1 you may
announce one of exactly **four** things, or stay silent:

| Announcement | Means | Value |
|---|---|---|
| **Tierce** | 3-card sequence | 20 |
| **Fifty** | 4-card sequence | 50 |
| **Hundred** | 5+-card sequence | 100 |
| **4x** | a four-of-a-kind | **100–200, not yet disclosed** |

No suit, no rank, no cards. Note the asymmetry: the three sequence classes each pin a value
exactly, but **`4x` spans 100 to 200** — a carré of queens and a carré of jacks are the same
announcement. Announcing `4x` therefore conceals a 2× spread in what you are claiming.

**Stage 1b — questioning. [OWNER]** An opponent *who holds a combination themselves* may ask you
to narrow the claim, and you must answer:

| For a | You answer |
|---|---|
| sequence (Tierce / Fifty / Hundred) | its **top card**, and **whether it is in the trump suit** |
| `4x` | its **rank** |

**You do not reveal the suit** — only the top card and a trump/not-trump bit. So a defender who
hears "Fifty, to the King, not trump" knows the rank and that it is one of the three non-trump
suits, but not which. That is materially less than naming the suit, and it is deliberately less:
knowing *which* non-trump suit an opponent has four cards of would be worth a great deal in the
play phase.

This is the second information release, and a bluffer who announced `4x` and is asked the rank
must commit to a specific one.

**You may announce a class you do not hold. [OWNER]** Nothing prevents a pure bluff: since showing
is a separate act, you can claim `4x` holding nothing. The only way opponents can punish it is to
announce their own real combination and wait for you to fail to show (stage 2), at which point the
claim lapses to them.

**Stage 2 — show, at hand 2.** Before playing your card to trick 2 you must *show* the actual
cards to collect the points. Announcing alone scores nothing.

**Concealment therefore has two distinct forms:**

| Form | What it costs | What it buys |
|---|---|---|
| Never announce | the points | reveals nothing at all |
| Announce truthfully, then withhold at hand 2 | the points | opponents who believed themselves beaten may not have announced — and without an announcement they cannot claim later |
| Announce a class you don't hold | nothing you had | the same suppression, at no real cost, plus a false read on your hand |

Information released, in increasing order: **class** (hand 1) → **top card + trump bit, or rank**
(if questioned) → **the actual cards** (hand 2 show). Each step is a separate decision point, and
the agent can stop at any of them. Note that even the second step withholds the suit — only the
show discloses it.

**Lapsed claims pass to the runner-up. [OWNER]** If a player with a smaller combination *also
announced* at hand 1, and the leading claimant then fails to show at hand 2 — by forgetfulness or
by design — the runner-up may show their own combination **before they play their next card**.
That is their hand-2 card if they have not yet played to trick 2, otherwise their hand-3 card. So
the window depends on turn order within trick 2, not on a fixed trick number.

A runner-up who successfully shows takes the comparison outright: **their team scores all of its
combinations**, exactly as if it had held the best one all along. **[OWNER]**

**Why this is worth modelling rather than simplifying away.** A shown combination is the single
largest public information leak in the game — showing a Hundred in spades reveals five of your
eight cards to three other players, right before the play phase that those cards have to survive.
So the real trade is *points for information*, and it produces a genuine two-sided interaction:

- A player holding a big combination and a fragile hand may announce (to suppress opponents'
  announcements) and then withhold (to protect the information). Both sides score nothing, which
  is worse for them in raw points than showing — but the hand stays hidden.
- Because bluffing is free, a player holding *nothing* can run the same suppression play at no
  cost whatsoever.
- The counter is to **announce anyway even when you believe you are beaten**, purely to retain the
  option of claiming if the leader withholds. Which means a truthful big holder is punished for
  the bluff being available at all.

That is a small, complete bluffing sub-game — costless false claims, a suppression motive, and a
defensive counter-announcement — embedded inside a trick-taking game and coupled to the play phase
through the information it leaks. It is one of the more novel things this environment offers and
belongs in the paper, not a footnote. Note that it also makes the game **not** a pure
imperfect-information card game with public actions: announcements are cheap talk with partial
commitment, which is closer to poker's bet-as-signal than to Bridge's conventions.

**Staging decision for the environment** (implementation choice, not a rules claim): M1 ships with
`declarations_are_actions: false` — every combination auto-detected, auto-announced and auto-shown
at trick 1. This is a **deliberate, documented deviation** that removes the entire sub-game above,
and it must be reported as such rather than quietly assumed. The reason to stage it is cost: the
protocol adds a contingent, turn-order-dependent action window that the information-leakage tests
(env spec §3) have to model carefully, and getting it wrong silently corrupts every result. It is
worth far more as a clean ablation — *does self-play discover announce-then-withhold, and does it
discover the defensive counter-announcement?* — than as noise in the first learning run.

---

## 7. Scoring — **this is where Blot Star differs most from classical Belote**

Classical Belote: each team simply banks the points it collected. **Blot Star instead pays out the
announced bid on top**, and pays the whole deal to the defenders on failure. Getting this wrong
invalidates every trained bidder, so it is stated in full.

### 7.1 The published formulas **[BS]**

**Trump contract**
- Success: `announced bid + collected points + bonuses`
- Lost: `announced bid + 16 points + bonuses` (to the defending team)

**No trump ("boy") contract**
- Success: `announced bid × 2 + collected points + bonuses`
- Lost: `announced bid × 2 + 16 points + bonuses`

**With Contra**
- Trump: `bid × 2 + 16 points + bonuses`
- No trump: `bid × 3 + 16 points + bonuses`

**With Recontra**
- Trump: `bid × 4 + 16 points + bonuses`
- No trump: `bid × 5 + 16 points + bonuses`

### 7.2 Reading the formulas as an algorithm

Bid multiplier is a **lookup table, not a formula** — note that no-trump is *one more* than trump
at each doubling level, which no single multiplicative rule reproduces:

| | no double | contra | recontra |
|---|---|---|---|
| **Trump** | ×1 | ×2 | ×4 |
| **No trump** | ×2 | ×3 | ×5 |

```
M          = bid_multiplier[contract_family][doubling]     # table above
bid_payout = M × announced_bid

# card portion. capot SETS the winner's card portion to 252, it does not add 90.  [OWNER]
if attackers took all 8 tricks:   cards_A, cards_D = 252, 0
elif defenders took all 8 tricks: cards_A, cards_D = 0, 252
else:                             cards_A = card_points(attackers) + says(attackers)
                                  cards_D = card_points(defenders) + says(defenders)
                                  # cards_A + cards_D == 162; exactly one team gets the 10 says

raw_A = cards_A + awarded_bonuses(attackers)               # §6.4 decides who is awarded bonuses
raw_D = cards_D + awarded_bonuses(defenders)

# combination bonuses COUNT toward fulfilment — this is what makes bids above 16 possible. [OWNER]
made  ⇔  raw_A ≥ 10 × announced_bid
         AND (attackers took all 8 tricks  if  capot_flag  else True)      # [OWNER]

if made:
    score_attackers = bid_payout + scale(raw_A)
    score_defenders = scale(raw_D)                          # defenders DO bank theirs  [OWNER]
else:
    score_attackers = 0
    score_defenders = bid_payout + 16 + scale(awarded_bonuses(defenders))
```
where `scale(x) = round10(x)` per §7.4. Fulfilment is **`≥`** — collecting exactly the bid makes
the contract **[OWNER]**.

On a failed contract where the **defenders** took all 8 tricks, the 16 in the payout becomes
**25** **[OWNER]**: `score_defenders = bid_payout + 25 + scale(awarded_bonuses(defenders))`.

**A capot bid has two independent failure modes.** `made` requires the point target *and* the
shutout. The owner's phrasing: bid `27 Capot`, collect 27 points, but let the opponents steal one
trick — **you still lose**. A plain bid of 27 with the same 27 points would have been made. This
is the sharpest decision in the game and the thing a bidding agent has to learn to be scared of.

Note the asymmetry that makes this ruleset interesting to bid in: a made contract pays
`M × bid` **on top of** the points actually collected, so overbidding is rewarded when it comes
off; a failed contract hands the defenders the entire deal (16) **plus** the bid. Under `NT` both
the reward and the penalty double. Contra/recontra scale only the bid term, never the 16 or the
collected points.

### 7.3 Capot **[OWNER]**

On a shutout the winning side's **card portion is set to 252 raw = 25 scaled**; the other side's
card portion is 0. Combination bonuses are then added on top of the 252 — which is how bids in the
30s and beyond become supportable.

Implement it as an assignment, not an increment:

```python
cards_winner = 252  # NOT collected + 90
cards_loser = 0
```

The two are numerically the same when the shutout side collected all 162, so a test that only
exercises the plain case will not catch an `+= 90` implementation. Test it against a deal that
*also* has bonuses.

**In the failed-contract branch, a defenders' capot replaces the 16 with 25. [OWNER]**

```
score_defenders = bid_payout + (25 if defenders took all 8 tricks else 16)
                             + scale(awarded_bonuses(defenders))
```

The 25 **replaces** the 16 — it does not stack on top of it. Concretely, an attacker who bids 14
in a trump suit and is then shut out loses `14 + 25 = 39` rather than `14 + 16 = 30`.

### 7.4 Scaling and rounding

Blot Star's "16 points" for a full deal fixes the scale at `raw / 10` (162 → 16). The exact
rounding rule is **[OPEN-10]**.

Default: `round10(x)` = divide by 10, round to nearest, **half rounds down** (`85 → 8`,
`86 → 9`, `77 → 8`, `155 → 15`).

**Correction (superseding an earlier draft of this doc).** An earlier version of this section
claimed that scaling both sides independently — `round10(cards_A) + round10(cards_D)` — always
equals 16, and instructed implementing it via a complement (`16 − other`) to guarantee that. Both
claims are **false**, and were caught by brute-force enumeration during M1 implementation, not
sourced from Blot Star or the owner: they were this document's own error.

`162` is not a multiple of `10`, so for *any* standard rounding rule (floor, ceiling, half-up,
half-down), two raw values that sum to 162 do not always have scaled values that sum to
`round10(162) = 16`. Exhaustively checking every split `x + (162−x) = 162` under half-down
rounding: **16 of the 163 splits violate the sum**, every one at `x ≡ 6 (mod 10)` (e.g.
`round10(106) + round10(56) = 11 + 6 = 17`) — and `(106, 56)` is exactly this document's own
§7.5 Fixture A card-points split, which makes the error easy to catch: reproducing that fixture's
worked numbers (§7.5) is only possible with **independent** rounding, not the complement. The
worked fixtures are correct as written; it is this section that was wrong.

**The actual rule, matching §7.2's pseudocode and every worked fixture in §7.5:** `round10` is
applied **independently** to each side's bonus-inclusive raw total (`raw_A`, `raw_D`), with no
complement step and no cross-side adjustment. There is no invariant that the two scaled scores
sum to a fixed constant — Fixture A's own `13 + 11 = 24` (not 16) is the proof, since combination
bonuses are included in what gets rounded and are not required to balance between sides at all.

The only invariant that **does** always hold, and is worth asserting in tests: absent a capot,
`cards_A + cards_D == 162` **raw** (unscaled) — the sum of the two sides' bonus-inclusive card
portions before rounding, not after.

### 7.5 Worked example (commit this as a test fixture)

Trump ♥, attackers bid **11**, no doubling. Attacking team = seats {0, 2}.
- Attackers take 96 card points and win the last hand: `96 + 10 = 106` raw.
- Seat 0 holds ♥K + ♥Q → Blot-Reblot 20, scored unconditionally (§6.2).
- Seat 1 (defender) holds a Fifty in ♠ (50); seat 2 (attacker) holds a Tierce in ♦ (20).
- Comparison: 50 > 20, neither is in trump → **defenders score their 50**; the attackers' Tierce
  scores nothing. The attackers' Blot-Reblot still counts.

```
raw_A = 106 + 20            = 126
raw_D = (152 - 96) + 50     = 106
made? 126 ≥ 110             → yes
score_attackers = 1×11 + round10(126) = 11 + 13 = 24
score_defenders =        round10(106) = 11
```
Same deal, attackers instead collect only 100 raw (fail):
```
score_attackers = 0
score_defenders = 1×11 + 16 + round10(50) = 11 + 16 + 5 = 32
```
Same failure under `NT` with the same bid: `2×11 + 16 + 5 = 43`.

**Fixture B — a high bid carried by combinations.** Trump ♠, attackers hold a carré of jacks (200)
and a Fifty (50) = 250 raw in bonuses, and bid **26**. They win no capot but collect 40 card
points and the last hand:
```
cards_A = 40 + 10 = 50
raw_A   = 50 + 250 = 300
made? 300 ≥ 260 → yes
score_attackers = 1×26 + round10(300) = 26 + 30 = 56
score_defenders = round10(162 - 50)   = round10(112) = 11
```
This is why the ladder runs well past 16, and why `NORM` in the reward function cannot be tuned
off the plain-deal range alone.

**Fixture C — the capot trap.** Same holding, attackers bid **27 Capot**. They collect 27 scaled
points comfortably but the defenders steal trick 6:
```
made? point target met, but capot_flag set and shutout failed → NOT made
score_attackers = 0
score_defenders = 1×27 + 16 + 0 = 43
```
Compare: a plain `27` on the identical deal would have been made. Commit all three fixtures.

---

## 8. Match end **[BS]**

- Target: **101, 151 or 301** — all three are real and the engine must support all three
  **[OWNER]**. Note 151, not the 201 quoted by other platforms.
- First team to reach the target wins.
- Both teams cross in the same deal → **higher total score wins** **[OWNER]**.

**Training default: 101**, chosen for episode length rather than authenticity — all three targets
are equally legitimate, so this is a compute decision and should be stated as one. Also support
**single-deal episodes** (`episode_unit="deal"`) as a first-class mode **[OWNER]**: it is the
cheapest signal for the trick-play policy, though it necessarily discards the endgame bidding
behaviour that the match target creates. Train on deals, evaluate on matches, and report both.

At 101 the target is small relative to what a deal can pay — an ordinary made contract is 16–32,
but a combination-carried deal like §7.5 fixture B pays 56 — so a match is often only 3–6 deals
and a single big hand can end it. **Match score is therefore a first-class part of the state.**
Bidding at 95–20 is a different problem from bidding at 0–0, and the observation must contain it.
At 301 that pressure relaxes and per-deal skill dominates, which is why the target belongs in the
ablation table rather than being silently fixed.

---

## 9. Complexity notes relevant to the RL design

- Distinct deals: `32! / (8!)^4 ≈ 9.96 × 10^16`; `≈ 4.15 × 10^15` up to suit permutation.
- Information set at the start of play: the other 24 cards split 8/8/8 →
  `24!/(8!)^3 ≈ 9.5 × 10^9` worlds, shrinking fast as voids are revealed.
- Play phase: ≤ 8 legal moves per node, 32 plies. Small enough that **exact double-dummy
  (perfect-information) solving is tractable** with alpha-beta + transposition tables. This is a
  major lever — see the environment spec §8.
- Auction: 5 contract types × levels 8–80 × a capot flag. Raw branching is large but the legal set
  is small — seniority is purely numeric, so a bid must strictly exceed the standing level and
  legality collapses the space to `(80 − current_level) × 5 × {1 or 2}` plus pass/contra/recontra.
  Auctions are typically 4–12 actions deep. Because passing is **not** binding, the auction can in
  principle circle many times, so the engine needs a hard step cap as a safety net.

---

## 10. Configuration flags

```yaml
rules:
  preset: "blotstar"

  # deal & seating                                    # [OWNER] confirmed
  deal_all_eight_before_auction: true                 # [BS]
  dealer_rotation: "clockwise"
  opener_and_leader: "dealer_left"                    # opens the auction AND leads trick 1

  # contracts                                          [BS] — no all-trump
  contract_types: ["C", "D", "H", "S", "NT"]

  # auction                                            # [OWNER] except where noted
  min_bid: 8
  max_bid: 80                                          # NOT 16; = capot 25 + carrés 20+14+11+10. §4.2
  bid_increment: 1
  bid_axis: "scaled"                                   # bid 12 == 120 raw
  capot_is_a_bid_modifier: true
  capot_sticky_upward: true                            # once bid, all higher bids must be capot
  capot_outranks_at_equal_level: false                 # 26 Capot does NOT beat plain 26; go to 27
  pass_is_binding: false                               # a passed player may bid again later
  nt_outranks_suit_at_equal_level: false               # seniority is purely numeric
  three_passes_after_bid: "close"                      # [OPEN-2c] "close" | "last_bidder_may_raise"
  max_auction_steps: 64                                # safety net: passing is non-binding
  contra_out_of_turn: false

  # play                                               # [OWNER] all confirmed
  must_follow_suit: true
  must_beat_when_trump_led: true
  must_ruff_when_opponent_winning: true
  must_overtrump: true
  must_ruff_when_partner_winning: false                # partner-winning exemption applies

  # combinations
  tierce: 20                                           # [BS]
  fifty: 50                                            # [BS]
  hundred: 100                                         # [BS]
  long_run_value: 100                                  # 6/7/8-card runs too  [OWNER]
  blot_reblot: 20                                      # [BS]
  blot_reblot_exempt_from_comparison: true             # never clashes; flat +20 to holder [OWNER]
  blot_reblot_in_notrump: false                        # no trump suit to hold K+Q of  [OWNER]
  carre_trump:   {J: 200, 9: 140, A: 110, 10: 100, K: 100, Q: 100, 8: 0, 7: 0}   # [BS]
  carre_notrump: {J: 100, 9: 0,   A: 190, 10: 100, K: 100, Q: 100, 8: 0, 7: 0}   # [BS]
  only_winning_team_scores_combinations: true          # [OWNER]
  combination_tie_resolution: "elder_hand"             # from the trick-1 leader  [OWNER]
  carre_always_beats_sequence: true                    # by category, not value  [OWNER]

  # announce / show protocol  §6.5                     # [OWNER]
  declarations_are_actions: false                      # M1 staging choice, NOT the real rule
  announce_classes: ["tierce", "fifty", "hundred", "4x"]   # a CLASS, never a value or a card
  announce_requires_holding: false                     # pure bluffs are legal and free
  questioning_enabled: true
  questioner_must_hold_combination: true
  question_discloses:                                  # NEVER the suit
    sequence: ["top_card", "is_trump"]
    carre:    ["rank"]
  lapsed_claim_passes_to_runner_up: true               # runner-up scores ALL its team's combos
  sequence_precedence: ["length", "top_card", "trump", "elder_hand"]   # [OWNER] top card outranks trump

  # scoring                                            # [BS] unless noted
  bid_multiplier:
    trump:   {none: 1, contra: 2, recontra: 4}
    notrump: {none: 2, contra: 3, recontra: 5}
  full_deal_scaled: 16
  fulfilment_strict: false                             # ">=" — exactly the bid makes it  [OWNER]
  combinations_count_for_fulfilment: true              # [OWNER] — this is what allows bids > 16
  defenders_score_collected_on_success: true           # [OWNER]
  capot_card_portion: 252                              # SET, not += 90      [OWNER]
  capot_bid_requires_shutout: true                     # point target alone is not enough [OWNER]
  failed_contract_base: 16                             # becomes 25 on a defenders' capot [OWNER]
  failed_contract_base_on_defender_capot: 25           # REPLACES the 16, does not stack  [OWNER]
  rounding: "half_down"                                # [OPEN-10]

  # match                                              # [OWNER]
  match_target: 101                                    # 101 | 151 | 301 — all three legitimate
  episode_unit: "deal"                                 # "deal" | "match" — both first-class
  tiebreak_on_double_cross: "higher_score"
```

Commit this as `presets/blotstar.yaml`, expose a stable `rules_hash`, and stamp that hash into
every checkpoint and every result table. Rule drift between runs is the single most likely way
this project produces unpublishable numbers.

---

## 11. Confirmed — quick reference

Lock all of this into tests immediately. **[BS]** = Blot Star page, **[OWNER]** = project owner.

**Structure**
- 4 players, 2 teams, 32 cards, 8 cards each, bid then play **[BS]**
- Exactly two contract families: trump suit, or no-trump ("boy"). **No all-trump.** **[BS]**
- A bid is a triple `(level, type, capot_flag)` **[OWNER]**

**Cards**
- Trump order `J 9 A 10 K Q 8 7` = `20 14 11 10 4 3 0 0` **[BS]**
- No-trump order `A 10 K Q J 9 8 7` = `19 10 4 3 2 0 0 0` **[BS]**
- Side suits in a trump contract use `A 10 K Q J` = `11 10 4 3 2` **[OWNER]**
- Last hand ("says") = 10, in both contract families **[OWNER]**

**Combinations**
- Tierce 20, Fifty 50, Hundred 100 — sequences in **natural** rank order **[BS]**
- Blot-Reblot = K+Q of trump = 20 **[BS]**; trump contracts only, none in no-trump **[OWNER]**
- Carré trump: J 200, 9 **140**, A **110**, 10/K/Q 100, 8/7 0 **[BS]**
- Carré no-trump: A **190**, 10/K/Q/J 100, 9/8/7 **0** **[BS]**
- Precedence **[OWNER]**: **any carré beats any sequence** (by category, not value) → carrés by
  value → sequences by **length → top card → trump → elder hand**. Worked example:
  `A-K-Q` non-trump > `J-10-9` trump > `J-10-9` non-trump
- Winning team scores **all** its combinations, losing team scores **none** **[OWNER]**
- **Blot-Reblot is exempt from the comparison entirely** — flat, unconditional +20 to its holder,
  and it does not win the comparison for its team either **[OWNER]**
- Runs of 6/7/8 cards are still 100 **[OWNER]**
- **Claiming is staged: announce a class at hand 1 → answer questions → show at hand 2.**
  Announcing alone scores nothing **[OWNER]**
- The four announcements are **Tierce / Fifty / Hundred / 4x** — no suit, no rank, no cards. `4x`
  hides a 100–200 spread **[OWNER]**
- An opponent **who holds a combination** may question you; you answer the **top card + whether
  it's trump** (sequence) or the **rank** (`4x`). **Never the suit** **[OWNER]**
- **Bluffing is free** — you may announce a class you don't hold, since showing is separate **[OWNER]**
- If the leader fails to show, a runner-up who also announced may show before their next card, and
  their team then scores **all** its combinations **[OWNER]**
- **Combinations count toward contract fulfilment** — this is what makes bids above 16 possible **[OWNER]**

**Play obligations** — all **[OWNER]**
- Must follow suit; must beat within trump when trump is led
- Must ruff when an opponent is winning and you hold trump; must over-trump when able
- **Exempt** from ruffing when your partner is winning the trick

**Auction** — all **[OWNER]**
- Opens at 8, raises by **+1** minimum, **not capped at 16** — combinations and capot support much
  higher bids; the maximum makeable bid is **80**
- **Seniority is purely numeric**: `NT` does not beat a suit at equal level, and capot does not
  beat a plain bid at equal level (`26 Capot` cannot answer `26` — go to `27 Capot`)
- Capot is a bid modifier and is **sticky upward**: once bid, every higher bid must also be capot
- **Passing is not binding** — a passed player may bid again later

**Scoring**
- Made: `M×bid + collected + bonuses`; failed: `M×bid + 16 + bonuses` to defenders **[BS]**
- Fulfilment is **`≥`** — collecting exactly the bid makes it **[OWNER]**
- Bid multiplier table: trump 1/2/4, no-trump 2/3/5 for none/contra/recontra **[BS]**
- Defenders **do** bank their collected points on a made contract **[OWNER]**
- Capot **sets** the card portion to 252, it does not add 90 **[OWNER]**
- A capot bid needs the point target **and** the shutout; either alone fails **[OWNER]**
- On a failed contract where the **defenders** shut them out, the 16 becomes **25** (replaces, does
  not stack) **[OWNER]**
- Match target 101 / 151 / 301, all three supported; both cross → higher score wins **[OWNER]**

---

## 12. Open questions

**Two remain, both deferred by the owner. Neither blocks M1.**

| # | Question | Default | Impact |
|---|---|---|---|
| **2c** | After 3 passes following a bid, does the auction close, or may the **last bidder** raise or change their own contract? gambler.ru describes the latter; Blot Star is silent. | `"close"` | **Structural.** If the last bidder may raise, a player can climb the ladder unopposed, which changes the auction game tree and reduces how much single-action bid expressiveness matters (env spec §3.1). Isolate it behind `three_passes_after_bid` so switching is a config change. |
| **10** | Rounding: does raw 85 scale to 8 or 9? | `half_down` | ~10% of deals, ±1 point on whichever side lands on a `x5` remainder. Best resolved from real deal lines (§13) rather than from memory. (Neither `half_down` nor `half_up` makes the two sides' scaled totals sum to a constant — see the §7.4 correction — so this choice does not carry the significance an earlier draft claimed for it.) |

### Resolved

| # | Question | Answer |
|---|---|---|
| ~~1~~ | Dealer rotation, opener, leader | Clockwise; dealer's left opens the auction **and** leads trick 1. §2 |
| ~~2~~ | Ladder: increment, pass re-entry, `NT` at equal level | +1; **passing is not binding**; `NT` does **not** outrank a suit at equal level. §4.2 |
| ~~2a~~ | Is the ladder capped at 16? | **No** — combinations count toward fulfilment. §4.2 |
| ~~2b~~ | Maximum bid | **80** = capot 25 + carrés J 20 + 9 14 + A 11 + 10 10. §4.2 |
| ~~3~~ | Play obligations | Beat within trump when trump is led; ruff when an opponent is winning; over-trump when able; **exempt when partner is winning**. §5 |
| ~~4~~ | Runs of 6, 7, 8 cards | **100**, same as a five-run. §6.1 |
| ~~5~~ | Losing side's combinations; ties | Losing side scores **none**; equal and neither in trump → **elder hand** from the trick-1 leader. Blot-Reblot is **exempt from the comparison entirely**. §6.4 |
| ~~6~~ | Defenders' capot on a failed contract | The 16 becomes **25** — replaces, does not stack. §7.3 |
| ~~6a~~ | Is capot `+90` or a set value? | **Sets** the card portion to 252. §7.3 |
| ~~7~~ | May a player conceal a combination? | **Yes** — and it's a two-stage announce/show protocol with free bluffing and lapsed claims passing to the runner-up. A real rule; staged out of M1 as a documented deviation. §6.5 |
| ~~8~~ | Fulfilment `≥` or `>` | **`≥`** — exactly the bid makes it. §7.2 |
| ~~9~~ | Do defenders bank collected points on a made contract? | **Yes.** §7.2 |
| ~~11~~ | Both teams cross the target same deal | **Higher total score wins.** §8 |
| ~~12~~ | `26 Capot` over a standing plain `26`? | **No** — seniority is purely numeric, go to `27 Capot`. §4.4 |
| ~~13~~ | Hundred vs carré of tens, both 100 | **Any carré beats any sequence**, by category not value. §6.4 |
| ~~14~~ | Higher top card, or trump? | **Top card outranks trump.** `A-K-Q` non-trump > `J-10-9` trump > `J-10-9` non-trump. §6.4 |
| — | Match target | **101, 151 and 301 all legitimate**; engine supports all three. Train on 101 for episode length — a compute decision, not a rules one. §8 |
| — | Side suits in a trump contract, bid axis, no Blot-Reblot in no-trump, last hand in no-trump | All four derivations **confirmed**. §3.2, §3.3, §4.2, §6.2 |

---

## 13. Sources

**Primary (authoritative for this project):**
- **[OWNER]** — the project owner, who plays the game. Overrides everything below, including
  Blot Star, wherever they conflict. Source of: the play obligations (§5), the uncapped ladder and
  capot-as-bid-modifier with sticky-upward behaviour (§4.2–4.3), combinations counting toward
  fulfilment, defenders banking collected points on a made contract, and capot *setting* rather
  than adding the card portion (§7.3).
- <https://blotstar.com/bazar-blot-rules/> — Blot Star official rules. Client-side rendered; the
  point tables are card *images*, and the orderings above were read from their alt text
  (`J, 9, A, 10, K, Q, 8, 7` for trump; `A, 10, K, Q, J, 9, 8, 7` for no-trump; the "Hundred"
  illustration is `7-8-9-10-J`).

**Secondary — used only to fill [OPEN] gaps, and superseded by Blot Star wherever they conflict:**
- <https://www.terz.am/rules.php?lang=ru> — Armenian platform, Russian-language rules. Source of
  the default bidding ladder (open at 8, +1, 4 passes → redeal) and play obligations.
- <https://www.gambler.ru/Bazar-belote> — Russian rules; agrees on the ladder and contra/recontra.
- <https://www.pagat.com/jass/belote.html> — classical Belote baseline; source of the default
  ruff/over-ruff obligations and the "only the highest combination's team scores" convention.
- <https://github.com/NikolayIT/BelotGameEngine> — reference C# engine (Bulgarian variant); useful
  for cross-validating a double-dummy solver, **not** for rules.

**Known conflicts, and who wins.** Precedence is `[OWNER]` > `[BS]` > secondary.

| Point | Secondary sources | Blot Star | Owner | Adopted |
|---|---|---|---|---|
| All-trump contract | exists | absent | — | **absent** |
| Carré of nines (trump) | 150 | 140 | — | **140** |
| Carré of aces (trump) | 100 | 110 | — | **110** |
| Capot | a 25-level bid (terz.am) | outcome worth 90 | **bid modifier; sets card portion to 252** | **owner** |
| Bid ladder maximum | 16 | unstated | **80** | **80** |
| Ruff when partner is winning | required (terz.am) | unstated | **exempt** | **exempt** |
| Combination tie, neither in trump | neither scores (classical) | unstated | **elder hand** | **elder hand** |
| Match targets | 201 / 301 | 101 / 151 / 301 | — | **101 / 151 / 301** |
| Deal payout | each team banks what it collected | **bid paid on top; whole deal + bid to defenders on failure** | — | **Blot Star** |
| Ruff when partner is winning | required (terz.am) | unstated | **exempt** | **exempt** |
