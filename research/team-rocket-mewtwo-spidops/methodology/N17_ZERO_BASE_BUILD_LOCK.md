# N17 Zero-Residual Base Build Lock

Status: **frozen before N17 model construction**  
Protocol: `rocket-n17-replicated-target-residual/v1`

- Protocol SHA-256:
  `0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4`.
- Target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.
- Builder SHA-256:
  `6bf0a48fdcc03336c032b640c5692261e4afa390c96bfafba88e42827f651938`.
- Source model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Source deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Source `net_v10.py` SHA-256:
  `130061f3a064c32b7f1496c2e11f472359686e9185dd42b7bf488b898f5b8762`.
- Residual-capable `net_v10.py` SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`.
- Adapter seed `817001`, rank 32, scale 1.0, exactly zero final
  projection, and exactly 13,088 adapter parameters.
- The output directory was absent when this lock was written.
- The builder compiled under Python 3.12.4 with PyTorch 2.13.0+cpu.

Locked command:

```powershell
python scripts/build_n17_zero_residual_base.py
```

The build has no strength or deployment authority. Before training, the new
loader/runtime must pass exact zero-residual model/deck/base-state audit and
full action/logit parity against the frozen live model on all frozen N6R and N9
support.
