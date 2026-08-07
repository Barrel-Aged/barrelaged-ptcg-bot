# BCv2 recovery V8 repair amendment

Frozen: 2026-08-06, after the complete first-arm gate was opened  
Final intended arena identity: `HelbrosRocketSpidopsBC2RecoveryV8`

## First-arm result

The preregistered zero-Wobbuffet third-Murkrow arm was healthy but failed its
broad admission gate:

- 14-10 directly against immutable V5;
- 14-10 directly against immutable V7;
- 0-4 against Ash Marnie;
- 0-4 against Froslass/Starmie;
- 4-0 against Alakazam.

All 60 completed games were terminal. The Soldier cell was not started because
the orchestration wrapper ended after the unusually slow Alakazam cell and the
already observed 0-4 Marnie and 0-4 Starmie results had independently triggered
the preregistered stop-for-repair rule. That arm is retained as development
evidence and will not be packaged or submitted.

## Diagnosis and final repair

The result reinforces the larger replay diagnosis: replacing Wobbuffet alone
does not solve the structural Marnie/Starmie deficit, while broader zero-
Wobbuffet rebuilds damaged Rocket density, openings, model alignment, or all
three. Wobbuffet's two bad Kaggle attacks are evidence for constraining its
attack role, not sufficient evidence for discarding the exact deck substrate
that produced BCv2's strongest external result.

The final V8 candidate therefore restores the byte-exact V5/BCv2 deck, model,
deck library, and packaged engine. It combines only four disclosed guards:

1. early Poke Pad search for a missing Spidops;
2. exact-lethal Brave Bangle attachment;
3. public-information Articuno fifth-slot reservation, including withholding
   the lone Wobbuffet from that last slot in a revealed effect-attack matchup;
4. the replay-exact Wobbuffet exit rule that redirects only selected Headbutt
   Bounce to an offered retreat when a Benched Spidops is already ready.

Rocket Mirror is never intercepted. The exit rule changed exactly the two bad
Headbutt decisions in the archived Kaggle replay audit and zero of 4,044 fresh
development decisions, so its action surface is known but its causal win value
remains unproven. No Erasure Ball override, search planner, hidden-hand read,
new training claim, or Kaggle submission is added.

## Final health gate

Before the only arena submission:

- verify exact BCv2 deck/model hashes and 60-card integrity;
- pass deterministic Articuno and Wobbuffet exit cases;
- complete a fresh one-game official-engine terminal smoke;
- complete 16 side-swapped direct games against V5 and 16 against V7;
- require every game terminal and at least four V8 wins in each direct cell.

This is deliberately a package-health and catastrophic-regression gate. The
internal arena is the requested external strength estimate; small local cells
will not be presented as proof of improvement. If the gate passes, package and
submit exactly `HelbrosRocketSpidopsBC2RecoveryV8`, then leave the identity
immutable and pause development.
