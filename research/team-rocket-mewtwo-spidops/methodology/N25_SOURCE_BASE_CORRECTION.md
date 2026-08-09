# N25 Source-Base Correction

Status: **frozen after the first builder failed closed and before any N25 model was written**  
Protocol: `rocket-n25-complementary-safe-composition/v1`

The original protocol correctly defines the action-head composition but
overstates the relationship between the serialized N8B `delta_100` arm and the
current live artifact. N8B arms were built on
`scale_value_calibrated_v1` (SHA-256
`cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`),
not directly on `leader_bc_scale_v2`.

The frozen N8B manifest proves:

- the N8B control and current live artifacts differ only in `value.*` tensors;
- their action-policy tensors, including both logit-head tensors, are exact;
- N8B `delta_100` differs from the N8B control in exactly
  `logit_head.weight` and `logit_head.bias`.

Therefore the N25 action delta remains exactly:

```text
delta_head = N8B_delta_100.head - N8B_control.head
           = N8B_delta_100.head - live.head
N25.head   = N15.head + delta_head
```

No N8B value tensor enters N25. Every non-head N25 tensor must remain exact N15.
The corrected builder must validate the N8B control hash, prove the
control/live value-only relationship, prove the delta/control two-head-tensor
relationship, and then prove the formula above bit-exactly after serialization.

The live, N8B control, and N8B delta artifacts must also have identical model
configuration dictionaries. N15 intentionally adds only
`adapter_rank: 16` and `adapter_scale: 1.0` to that shared configuration
because its frozen residual introduces `logit_adapter.*`; every inherited
architecture setting must remain identical. This metadata-only distinction is
required and does not relax any tensor-composition rule.

The initial builder stopped at this source audit before creating its output
directory, model, report, held-out metric, or official-engine game. This
correction changes no candidate, scale, gate, target, seed, panel geometry, or
deployment condition.
