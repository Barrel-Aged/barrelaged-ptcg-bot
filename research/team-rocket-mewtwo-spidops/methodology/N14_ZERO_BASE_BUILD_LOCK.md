# N14 Zero-Residual Base Build Lock

Status: **frozen before N14 model construction**  
Protocol: `rocket-n14-localized-contextual-residual/v1`

- Protocol SHA-256:
  `2cbf0206af920072705421a349b7d13476766a1eff61dbd0494844bd0ed6821d`.
- Builder SHA-256:
  `ed04a002246ebdf6d51e23155f525dfaa38b1f1786cdc483f62935f317d60a46`.
- Builder tests SHA-256:
  `616462337392f5333b4d3adeb839418fa409029b320c1eca5dccfda21c38b8bf`.
- Source model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Source deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Source `net_v10.py` SHA-256:
  `130061f3a064c32b7f1496c2e11f472359686e9185dd42b7bf488b898f5b8762`.
- Adapter seed `814001`, rank 16, scale 1.0, exactly zero final projection.

Locked command:

```powershell
python scripts/build_n14_zero_residual_base.py
```

The build has no strength or deployment authority. Before training, the new
loader/runtime must pass exact zero-residual model/deck/base-state audit and
full action/logit parity against the frozen live model.

