# N14 Localized Contextual Residual Result

Status: **complete negative offline result; no official match panel and no deployment authority**  
Protocol: `rocket-n14-localized-contextual-residual/v1`  
Completed: 2026-08-01

## Outcome

The zero-initialized, rank-16 contextual residual solved the intervention-capacity
problem seen in N12 and N13: it moved all 48/48 strict counterfactual corrections.
It did not solve the safety problem. None of the four preregistered first-hit
checkpoints preserved enough of the frozen live policy to enter an official-engine
development panel.

The live bot therefore remains exactly unchanged. N14 supplies no arena-strength
claim, no deployment authority, and no permission to inspect match outcomes.

## Frozen provenance

- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- frozen live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- protocol SHA-256:
  `2cbf0206af920072705421a349b7d13476766a1eff61dbd0494844bd0ed6821d`;
- zero-base model SHA-256:
  `0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb`;
- patched `net_v10.py` SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- zero-parity report SHA-256:
  `c8638bd6d5a879d32eb88bba30f7ddeb334d222e9b295c64f0ca2619096611d5`;
- training report SHA-256:
  `84ce98182af34919ba753a72be3d71242b31b0d290f16c29c4b77f7729511581`.

The zero-base audit reproduced the live top action on all 639 N6R roots and all
6,753 N9 actions, with zero relative-logit error. Every trained checkpoint also
had exact cached/full-runtime parity on both corpora. Only the four
`logit_adapter.*` tensors changed; all original network tensors remained frozen.

## Preregistered checkpoint results

| Checkpoint | Step | Corrections | N6 top-1 | N6 KL | N9 top-1 | N9 KL | V110 top-1 | Recovery top-1 | Admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fit12` | 62 | 12/48 | 85.448% | 0.125452 | 93.558% | 0.033832 | 93.403% | 93.722% | No |
| `fit24` | 90 | 24/48 | 87.310% | 0.120167 | 94.210% | 0.026653 | 94.184% | 94.237% | No |
| `fit36` | 149 | 37/48 | 89.340% | 0.086144 | 94.891% | 0.017540 | 94.907% | 94.874% | No |
| `fit48` | 286 | 48/48 | 93.739% | 0.043964 | 95.735% | 0.012000 | 95.457% | 96.027% | No |

The immutable checkpoint hashes are:

- `fit12`: `e498bac097594901e35ca5f4b57b83ea4b343bb21561abc91cd08fe267ffab0b`;
- `fit24`: `56a62c22a59f6f9710fee04173172eb9080ea0003b57e71b8d8f2fc5a4ac8c63`;
- `fit36`: `bea9104a72154c5a609ab0e06d193e3c48468781ca22af8cd219c6eb43571c49`;
- `fit48`: `0a11204beb06645334647211ca17af978023c864ac74367649e4d7d8451a361f`.

Admission required N6 top-1 retention at least 98%, N9 overall retention at
least 99%, each N9 opponent group at least 98.5%, N6 forward KL at most 0.05,
and N9 forward KL at most 0.02. `fit48` passed both KL ceilings and fit every
correction, but it still changed too many non-target top actions. All earlier
checkpoints failed even more preservation gates.

## Interpretation

N14 establishes two useful facts without implying strength:

1. A small nonlinear residual on the contextual option embedding has enough
   capacity to recognize every currently validated correction pattern.
2. The first time each fit threshold is reached is a transient, unsafe point;
   correction fit arrives substantially before baseline behavior is restored.

The locked optimizer continued through step 2,000 and still fit 48/48. Its
sampled minibatch KL values fell to about 0.0104 on N6R and 0.00447 on N9,
compared with the larger values near early checkpoints. This is only a training
trajectory observation: no terminal checkpoint was saved, full-corpus top-1
retention was not evaluated at step 2,000, and the observation cannot authorize
post-hoc checkpoint selection or match play.

## Next admissible experiment

A successor may preregister a distinct consolidation phase starting from the
immutable `fit48` checkpoint. It should treat the 48 corrections as hard margin
constraints while explicitly projecting the residual back toward the frozen
live policy on the complete preservation supports. Checkpoints must be selected
at fixed, preregistered consolidation boundaries rather than at first-fit times,
and the unchanged N14 offline gates must run before any official-engine match.

This is a new hypothesis and must receive a new protocol, seed, script hash,
output directory, and execution lock. N14 itself is closed.
