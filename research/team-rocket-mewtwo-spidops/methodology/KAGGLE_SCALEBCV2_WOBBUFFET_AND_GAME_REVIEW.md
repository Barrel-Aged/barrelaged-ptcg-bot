# HelbrosRocketSpidopsScaleBCV2: Wobbuffet validation and Kaggle game review

## Executive verdict

**Keep Wobbuffet provisionally as a one-copy tech, but do not validate it as a primary attacker.**

The user's observation found a real policy defect, but the replay frequency is smaller than the visual impression:

- Wobbuffet made **2 of 103 attacks (1.94%)**, both in one loss.
- Both attacks were `Headbutt Bounce`, both were chosen with a ready Spidops on the Bench and a legal retreat, and neither used Wobbuffet's distinctive `Rocket Mirror` attack.
- On the first decision, the exact submitted neural policy assigned **99.56%** probability to Headbutt Bounce and **0.44%** to retreat. This was learned policy behavior, not a legality failure or a display artifact.
- Wobbuffet opened Active three times, but all three were forced one-Basic hands. The policy never voluntarily chose it as the opening Active.

Wobbuffet is a real but minority competitive tech in Mewtwo/Spidops. It is not part of Juho Kallama's supplied 7th-place NAIC list, and it is not a core card in the current most-played variant. A direct swap to Honchkrow is not supported: Honchkrow is normally the centerpiece of a separate Porygon2/Rocket Feathers archetype. The strongest controlled deck-slot comparison is **one Wobbuffet versus one Team Rocket's Mimikyu**, with the policy retrained or fine-tuned for the changed deck.

The largest general improvement opportunity is not the Wobbuffet slot by itself. It is **preserving a next attacker and valuing future energy correctly**, especially when Erasure Ball can discard the only energy on a backup Spidops. The bot went 0-3 against Archaludon/Cinderace despite attacking and taking its first prize on global turn 3 in all three games.

## Audit scope and reproducibility

Cutoff: **2026-08-02 00:26 UTC** (2026-08-01 18:26 MDT).

- Kaggle submission: `55172263`, `HelbrosRocketSpidopsScaleBCV2.tar.gz`
- Status and public score at cutoff: **complete, 876.0**
- Replays: **all 20 completed episodes then available** — 19 public games and one validation mirror
- Public results visible in those replays: **13-6 (68.4%)**
- Including the validation mirror: **14-6 (70.0%)**
- Frozen deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- Frozen model SHA-256: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
- Deck integrity: the exact 60-card hash passed in every replay.
- Exact neural replay: **1,225/1,227 decisions matched byte-for-byte; the other two selected the same forced set in the opposite order. There were zero substantive mismatches.**

Machine-readable evidence:

- `work/kaggle_scalebcv2_55172263_20260801/audit.json`
- `work/kaggle_scalebcv2_55172263_20260801/audit.md`
- `work/kaggle_scalebcv2_55172263_20260801/policy_introspection.json`
- `work/kaggle_scalebcv2_55172263_20260801/turn_review_89414044.md`
- `scripts/audit_kaggle_scalebcv2.py`
- `scripts/inspect_kaggle_scalebcv2_policy.py`
- `tests/test_audit_kaggle_scalebcv2.py`

The audit uses global engine turn numbers. A win with fewer than six prizes can reflect a concession, deck-out, board-out, or another terminal condition, so prize counts should not be mistaken for the only win condition.

## The submitted deck is not the Juho list

The Kaggle deck is an arena/Kaggle-adapted ScaleBCV2 list:

### Pokémon — 15

- 4 Team Rocket's Tarountula
- 4 Team Rocket's Spidops
- 2 Team Rocket's Mewtwo ex
- 2 Team Rocket's Articuno
- 2 Team Rocket's Murkrow
- 1 Team Rocket's Wobbuffet

### Trainers — 31

- 4 Team Rocket's Ariana
- 4 Team Rocket's Transceiver
- 4 Poké Pad
- 3 Bug Catching Set
- 3 Team Rocket's Giovanni
- 2 Team Rocket's Proton
- 2 Energy Search
- 2 Team Rocket's Factory
- 2 Lillie's Determination
- 1 Team Rocket's Petrel
- 1 Team Rocket's Archer
- 1 Night Stretcher
- 1 Hero's Cape
- 1 Brave Bangle

### Energy — 14

- 7 Basic Grass Energy
- 3 Basic Psychic Energy
- 4 Team Rocket's Energy

It contains **no Honchkrow**. Juho's supplied NAIC list instead used two Mimikyu and one Lillie's Clefairy ex, with Ultra Ball, more Night Stretcher, Lucky Helmet, Prism Tower, Energy Switch, and Maximum Belt. It had no Murkrow, Wobbuffet, or Honchkrow. Wobbuffet is therefore only one part of a broader deck-engine departure, not an isolated one-card change.

## What the three attackers actually do

### Spidops

Charging Up accelerates a Basic Energy from the discard pile to each Spidops once per turn. Rocket Rush costs `{G}{C}` and does 30 damage for each Team Rocket Pokémon in play — **180 damage on the six-Rocket board present in episode 89414044**. It is the deck's efficient single-prize attacker and its energy engine.

### Wobbuffet

Rocket Mirror costs `{P}{C}` and moves all damage counters from a Benched Team Rocket Pokémon to the opposing Active. This can heal a damaged Benched asset while creating a knockout, but it requires a specific multi-turn state: a damaged Rocket must survive and reach the Bench, Wobbuffet must become Active, and its Psychic cost must be live.

Headbutt Bounce costs `{P}{C}{C}` for only 70 damage. It is a fallback or exact-prize attack, not a reasonable default over a ready six-Rocket Spidops.

Wobbuffet also supplies a Basic, single-prize Team Rocket body for Power Saver and Rocket Rush. That role is real even when it never attacks.

### Honchkrow

Honchkrow is a Stage 1 evolving from Murkrow. Rocket Feathers costs `{C}{C}` and does 60 damage per Team Rocket Supporter discarded from hand. Adding it is not merely selecting a better attacker from the current deck: it consumes a deck slot, an evolution turn, and future Supporter resources, and it changes what the policy must search for and conserve.

Tournament lists containing Honchkrow overwhelmingly classify as a separate Honchkrow/Porygon2 archetype rather than a Mewtwo/Spidops hybrid. It should be evaluated as a separate deck project, not as the immediate Wobbuffet fix.

## Competitive validation of Wobbuffet

The current Limitless average for the most popular Mewtwo variant, across 11 decklists, is:

- 4.00 Tarountula, 4.00 Spidops, 2.00 Mewtwo ex
- 1.91 Articuno, 1.82 Mimikyu, 1.00 Clefairy ex
- **0.27 Wobbuffet, 0.09 Murkrow, and no Honchkrow**

An average of 0.27 means Wobbuffet is a minority singleton tech rather than a core inclusion. It nevertheless has credible tournament precedent in Mewtwo/Spidops, including lists placing 9th at Los Angeles, 15th at Campinas, and 22nd at Utrecht, plus two NAIC 2026 lists. Sources:

- [Current Rocket's Mewtwo average card counts](https://limitlesstcg.com/decks/337/cards)
- [Tournament decklists containing Team Rocket's Wobbuffet](https://limitlesstcg.com/cards/DRI/82/decklists)
- [Tournament decklists containing Team Rocket's Honchkrow](https://limitlesstcg.com/cards/ASC/127/decklists)

This supports a narrow conclusion: **one Wobbuffet is defensible; building the attack plan around it is not.**

## Wobbuffet in all 20 Kaggle games

| Measure | Result |
|---|---:|
| Copies in deck | 1 |
| Forced Wobbuffet openings | 3/20 |
| Record in those openings | 1-2 |
| Games seen in play | 8/20 |
| Voluntary plays from hand | 5 |
| Attachments to Wobbuffet | 7 |
| Wobbuffet attacks | 2/103 (1.94%) |
| Games containing a Wobbuffet attack | 1/20 |
| Rocket Mirror uses | 0 |
| Wobbuffet attacks over ready Spidops with legal retreat | 2/2 |

All seven attachments were made while Wobbuffet was Active. Six were the two attachments made during each forced Wobbuffet opening; the seventh came after Wobbuffet was promoted late in the collapsing Archaludon/Cinderace game 89415634. It was not siphoning routine Bench attachments across the sample.

With 11 Basic Pokémon in the deck, a one-copy Wobbuffet is the only Basic in about **4.66% of otherwise valid random opening hands**. Three forced starts in 20 games is an unlucky but plausible small-sample cluster, and it helps explain why Wobbuffet looked more central in the visual replays than its 1.94% attack share indicates.

### Exact policy evidence from episode 89414044

At global turn 8, Wobbuffet had Grass plus Team Rocket's Energy, and a two-Grass Spidops was ready on the Bench. The legal choices and exact neural probabilities were:

| Choice | Probability |
|---|---:|
| Headbutt Bounce | **99.558%** |
| Retreat | 0.442% |
| End turn | 0% (suppressed by the attack-safe guard) |

At global turn 10, with the same ready Spidops:

| Choice | Probability |
|---|---:|
| Headbutt Bounce | **53.552%** |
| Team Rocket's Giovanni | 9.801% |
| Retreat | 8.662% |
| Other legal attachments/actions | remaining probability |

The two 70-damage attacks eventually knocked out Shaymin, after which Wobbuffet was knocked out by Crustle. A six-Rocket Spidops could instead attack for 180. That makes the first Wobbuffet attack a high-confidence tempo error. It does **not** prove the unplayed branch would have won the entire game, because the Kaggle replay cannot be restored and branched through the official engine.

## Larger official-arena context

The independent official-engine autopsy contains 4,880 ScaleBCV2 games, with a 2,724-2,154-2 record (55.8%). Its Wobbuffet associations are:

| Slice | Games | Record | Win rate |
|---|---:|---:|---:|
| All ScaleBCV2 games | 4,880 | 2,724-2,154-2 | 55.8% |
| Wobbuffet opening Active | 422 | 220-202 | 52.1% |
| Wobbuffet opening and later attacked | 38 | 16-22 | 42.1% |
| Wobbuffet opening and never attacked | 384 | 204-180 | 53.1% |
| Any Wobbuffet attack | 50 | 19-31 | 38.0% |
| Any Rocket Mirror | 18 | 4-14 | 22.2% |
| Any Headbutt Bounce | 33 | 15-18 | 45.5% |

There were only 94 Wobbuffet attacks in 4,880 games: 25 Rocket Mirrors and 69 Headbutt Bounces. These results are observational and confounded by bad starts and losing states, but they reinforce the tactical conclusion: **Wobbuffet is much healthier as an occasional body/tech than as a default attacker.**

The broader data does not prove a universal deck cut. After stratifying by opponent family and turn order, Wobbuffet openings were slightly positive relative to other starts in Marnie games but materially worse in Rocket mirrors and several other families. A matchup-conditioned policy or controlled deck A/B is more defensible than a blanket removal based on raw win rate.

## General Kaggle findings

### Matchups

| Opponent | Record |
|---|---:|
| Alakazam | **5-0** |
| Mega Lucario | **3-0** |
| Pure Archaludon | 1-0 |
| Dragapult | 1-0 |
| Great Tusk/Crustle | 1-0 |
| Crustle | 1-0 |
| Marnie/Grimmsnarl/Froslass | **1-2** |
| Archaludon/Cinderace | **0-3** |
| Kangaskhan/Crustle | 0-1 |
| Rocket mirror (validation, not public) | 1-0 |

The small Kaggle sample strongly identifies Archaludon/Cinderace as the immediate external matchup hole. The Marnie result agrees with the much larger official-arena sample, where ScaleBCV2 was 873-1,315-2 across 2,190 Marnie-family games (39.9%).

### Attack distribution

| Attacker | Attacks | Share |
|---|---:|---:|
| Team Rocket's Mewtwo ex | 49 | 47.6% |
| Team Rocket's Spidops | 40 | 38.8% |
| Team Rocket's Tarountula | 7 | 6.8% |
| Team Rocket's Murkrow | 5 | 4.9% |
| Team Rocket's Wobbuffet | 2 | 1.9% |

The bot is already primarily a Mewtwo/Spidops attacker. Wobbuffet is a narrow outlier decision, not the global attack identity.

### What is working

- Every game reached six Team Rocket Pokémon in play. Power Saver and Rocket Rush setup width are reliable.
- There were no illegal choices and no turns ended while a legal attack was available. The attack-safe deployment guard worked.
- All 20 exact replays matched the packaged model's behavior, so the analysis is about the deployed bot rather than a reconstructed approximation.
- Murkrow opened Active in six games and went 6-0; it used Deceit five times across the panel. This is small-sample evidence, but it argues against removing the unusual Murkrow consistency package without a controlled test.
- Alakazam and Mega Lucario were handled cleanly. Those families must be regression gates for any change.

### What is failing

Four of six public losses never had two legal attackers ready at the same time, and three of six never converted Mewtwo into an attack. The larger arena audit also found no second ready attacker in 1,234 of 2,154 losses. Short wins often ended before a backup was needed, so this is not a rule to maximize two ready attackers in every state; it is a warning for matchups that can return-knock out the current attacker.

The three Archaludon/Cinderace losses are especially consistent:

- All three reached six Rocket Pokémon.
- All three attacked and took the first prize on global turn 3.
- All three then lost by wide prize margins: 1-5, 2-5, and 1-5.
- None ever had a second ready attacker.
- In two replays, Erasure Ball discarded energy from the only developing Benched Spidops immediately before the Active Mewtwo was knocked out. The promoted Spidops then had to rebuild from zero.

This is a strategic resource-valuation error, not a setup-speed problem. Erasure Ball's optional 60/120 extra damage must be valued against the probability that the discarded energy is the next attacker's entire tempo. The correct choice depends on whether the extra damage crosses an exact knockout/prize threshold and whether the Active Mewtwo is likely to survive.

## Per-game review

| Episode | Result | Opponent | Opening | Review |
|---:|:---:|---|---|---|
| 89409698 | W (validation) | Rocket mirror | Murkrow | Full 5-5 mirror with Spidops, Mewtwo, and Murkrow attacks; second attacker ready by turn 11. Useful parity check, not an independent public result. |
| 89409774 | W | Mega Lucario | forced Tarountula | Mewtwo carried four attacks after an early Tarountula attack; 5-2 prize state. Strong result, though only one attacker was ever ready at a time. |
| 89410307 | W | Marnie/Froslass | Murkrow | Murkrow setup converted to four Spidops attacks and a 5-2 prize state. Slow first attack on turn 6 and no simultaneous backup, but the line worked. |
| 89410836 | W | Alakazam | Tarountula | One Tarountula attack and a 0-0 prize terminal. Treat as terminal-condition evidence, not proof of a completed prize plan. |
| 89411061 | W | Mega Lucario | Murkrow | Close 4-4 state using three Spidops and two Mewtwo attacks. Good attacker transition despite no simultaneous second-ready snapshot. |
| 89411365 | W | Alakazam | Tarountula | Long recovery game using Tarountula, Spidops, Mewtwo, and Articuno as an Active pivot. A second attacker arrived only on turn 20, but the control line held. |
| 89411898 | L | Marnie/Froslass | forced Tarountula | Closest loss at 5-5. Four Spidops attacks; Mewtwo appeared but never attacked. Priority is final-prize mapping and converting the developed Mewtwo line, not Wobbuffet. |
| 89412441 | W | Great Tusk/Crustle | forced Wobbuffet | Correctly avoided Wobbuffet attacks, eventually transitioning to Spidops and Mewtwo. The win shows the slot is playable, while the turn-7 first attack shows the two-retreat-cost opening tax. |
| 89412964 | W | Dragapult | forced Tarountula | Fast turn-3 pressure and a 5-2 prize state with Tarountula, Spidops, and Mewtwo all attacking. Preserve this tempo in regressions. |
| 89413500 | L | Marnie/Froslass | forced Wobbuffet | First attack only on turn 7, one Spidops attack total, no second ready attacker, and no Wobbuffet attack. This is an opening/energy-recovery collapse, not Wobbuffet over-attacking. |
| 89414044 | L | Kangaskhan/Crustle | forced Wobbuffet | The concrete Wobbuffet defect: two Headbutt Bounces over a ready Spidops and legal retreat; first attack turn 9. Highest-confidence tactical correction. |
| 89414572 | L | Archaludon/Cinderace | Tarountula | Took first prize on turn 3 but finished 1-5. Never established a second ready attacker; rebuilt successive Actives from scratch. |
| 89414576 | W | pure Archaludon | forced Tarountula | Three Spidops attacks and a 2-0 terminal. Do not combine this with the Cinderace variant; the matchup behavior is materially different. |
| 89415106 | W | Crustle | Murkrow | Two Deceits followed by a Spidops attack; 1-1 terminal. Supports Murkrow as a consistency opener rather than an Honchkrow promise. |
| 89415634 | L | Archaludon/Cinderace | forced Mewtwo | Mewtwo and Spidops both attacked, but never remained ready together. Erasure Ball consumed a developing Spidops's energy before the Mewtwo knockout; finished 2-5. |
| 89416179 | W | Mega Lucario | Murkrow | Turn-2 Deceit, turn-4 first prize, and an early 1-1 terminal. Efficient short-game setup. |
| 89416722 | W | Alakazam | Murkrow | Two Spidops attacks for a 2-0 terminal. Clean, compact line; no Wobbuffet involvement. |
| 89417296 | W | Alakazam | forced Mewtwo | Thirteen Erasure Balls with a 1-0 terminal. Durable Mewtwo control/deck-out style line, but it relied on one attacker throughout. |
| 89417876 | W | Alakazam | forced Mewtwo | Fifteen Erasure Balls and a 5-1 prize state. Strongest evidence that the single-Mewtwo plan is appropriate in this matchup. |
| 89418436 | L | Archaludon/Cinderace | forced Mewtwo | Third repeat of the matchup failure: first prize turn 3, final 1-5, no second ready attacker. A charged Benched Spidops again lost its energy to Erasure Ball before being promoted empty. |

## Prioritized improvements

### 1. Correct Wobbuffet's exit/attack valuation

Train or safely filter the precise state class rather than banning Wobbuffet attacks globally:

- If Wobbuffet is Active, retreat is legal, and Spidops or Mewtwo is already ready, heavily prefer the primary attacker.
- Permit Rocket Mirror when transferred damage creates a concrete knockout/prize swing or preserves a high-value damaged Benched attacker.
- Permit Headbutt Bounce only for an exact prize/tempo line that is at least as good as the ready primary attack.
- Do not attach the third attack energy to Wobbuffet without an explicit attack or escape plan.

Episode 89414044 and the larger 50-game Wobbuffet-attack arena slice are suitable relabel fixtures. A learned action-value correction is preferable; a narrow deterministic guard can be a deploy-time safety backstop after regression testing.

### 2. Add an Erasure Ball resource/optionality target

For every optional Bench-energy discard, score:

- damage without discarding, with one discard, and with two;
- whether each damage tier crosses an exact knockout/prize threshold;
- the survival probability of Active Mewtwo;
- whether the discarded card is the only energy on the only next attacker;
- the turns required to rebuild after the likely return knockout.

Against Archaludon/Cinderace, preserving a ready or one-turn-away Spidops should often outrank redundant damage. This is the cleanest place to use the project's resource-efficiency and optionality value heads.

### 3. Build an Archaludon/Cinderace-specific training/evaluation slice

The failure repeats three times with the same shape, so create replay fixtures around:

- the turn before the first Active knockout;
- every Erasure Ball discard selection;
- promotion choice after knockout;
- Charging Up allocation between Active and Bench;
- prize thresholds for Archaludon versus Cinderace.

The target is not a faster first attack; turn-3 pressure already exists. The target is a two-attacker prize chain that survives the return knockout.

### 4. Continue Marnie/Froslass work

The public 1-2 result agrees with the 2,190-game arena weakness. Separate the close 5-5 final-prize case from the forced-Wobbuffet setup collapse. Candidate labels should emphasize Mewtwo conversion, recovery after disruption, and avoiding a board that is wide but cannot produce a second attack.

### 5. Test Wobbuffet versus Mimikyu as a deck A/B

Do not edit the live/frozen bot based on 20 games. Run a controlled, side-swapped official-engine comparison:

1. exact ScaleBCV2 control;
2. Wobbuffet → Mimikyu, with deck-aware policy fine-tuning;
3. the same policy correction while retaining Wobbuffet.

Mimikyu is the strongest comparator because it is a Team Rocket Basic with free retreat, is near-core in current professional Mewtwo lists, and preserves Power Saver/Rocket Rush count. Test the full Juho-style Mimikyu/Clefairy/Ultra Ball engine separately; changing several engine cards and the Wobbuffet policy in one experiment would make the result uninterpretable.

### 6. Keep Honchkrow separate

If Honchkrow is desired, build and train a distinct Rocket Feathers candidate with Porygon2-style resource generation and explicit Supporter-discard planning. Do not treat `Wobbuffet -1, Honchkrow +1` as a validated patch to ScaleBCV2.

## Admission gate for a successor

A successor should not replace the frozen ScaleBCV2 identity until it passes all of the following on official-engine execution:

- zero legality or serialization regressions;
- no regression against Alakazam, Mega Lucario, Dragapult, and Rocket mirrors;
- improved paired performance against Archaludon/Cinderace and Marnie/Froslass;
- near-zero unjustified Wobbuffet attacks over a ready primary attacker;
- improved next-attacker readiness after Erasure Ball without losing required one-hit knockouts;
- separate reporting for policy-only and deck-change effects.

No Kaggle submission or unverified counterfactual performance claim was made as part of this review.

## Evidence boundaries

- Nineteen public games are enough to find repeated decisions, not enough to estimate a stable matchup win rate.
- The 4,880-game arena associations are larger but still observational; Wobbuffet often attacks from already-bad states.
- Exact logits establish why the deployed policy chose an action. They do not establish the terminal result of an unplayed branch.
- Competitive deck frequency validates plausibility, not optimality for this engine, action representation, and learned policy.
- The current high public score argues for controlled iteration, not an immediate deck rewrite.
