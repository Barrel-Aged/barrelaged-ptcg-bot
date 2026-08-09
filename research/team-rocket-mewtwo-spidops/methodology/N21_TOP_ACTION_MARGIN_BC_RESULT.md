# N21 Top-Action-Margin BC Result

Status: **rejected offline; no official-engine panel and no deployment**  
Protocol: `rocket-n21-top-action-margin-bc/v1`

## Integrity

- protocol SHA-256:
  `2a18a5f8ad3242e5d4c13ff132187121563b20b27f384597b91dc25d607cf839`;
- execution-lock SHA-256:
  `d90ab10547b8b57fd2b2a12788599ad695c547e9e6c2de6b9e4cd9f2973a5a1f`;
- trainer SHA-256:
  `22fdd063349f4a57732eb8607359db0707962dea2a466d2a5212ec67488de85e`;
- terminal model SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- training report SHA-256:
  `8f51c4ad18558ef7af48ca7a08fd9eb2e059a7e5f94290b0eb0a1f345357a0f6`;
- provenance SHA-256:
  `a53cda45e9ae2f74ea728b6d9a943b3c9c278b587398ef502a0f8740e3cbf73b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The sole locked command completed normally: two epochs, 198 optimizer steps,
seed `821001`, terminal checkpoint only. The saved model reloads byte-validly,
has no shape or finite-value errors, changes exactly the locked 14 tensors, and
leaves the other 45 tensors exact.

## Outcome

The model learned the held-out fresh-leader signal:

- teacher validation top-1: 76.30% live to **80.65%** (+4.35 pp), with loss
  0.7132 to **0.5945**;
- teacher test top-1: 77.59% live to **82.95%** (+5.36 pp), with loss 0.6617
  to **0.5506**.

It nevertheless failed the preregistered preservation gates:

| Gate family | Locked requirement | N21 terminal |
|---|---:|---:|
| Fresh live-top retention, train | >=99.5% | **99.23%** |
| Fresh live-top retention, validation | >=99.5% | **97.62%** |
| Fresh live-top retention, test | >=99.5% | **98.25%** |
| Recorded validation action | within 0.5 pp of live | **-2.38 pp** |
| Recorded test action | within 0.5 pp of live | **-1.50 pp** |
| N6 unchanged anchors | >=99.0% | **100.00%** |
| N9 overall | >=99.0% | **98.53%** |
| N9 V110 group | >=98.5% | **98.41%** |
| N9 RecoveryV4 group | >=98.5% | **98.67%** |

All KL gates passed (fresh 0.00154-0.00198, N6 0.00126, N9 0.00161),
demonstrating again that small mean KL does not guarantee preservation of the
deployed top action. N6 was fully preserved, but the broader fresh and N9
supports were not.

## Decision

`offline_admission=false`, `development_panel_authority=false`, and
`deployment_authority=false`. No build, runtime-parity audit, official-engine
match, confirmation, packaging, submission, or live mutation is authorized.
The live bot remains model
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

N21 closes coefficient-weighted top-action-margin BC as a rescue of the N19
direction. A successor must change the mechanism rather than tune these loss
weights or resample these splits. The useful surviving evidence is that a
leader-imitation candidate contains held-out signal while its disagreements
with the live Rocket policy must be selected state-conditionally, not applied
globally.
