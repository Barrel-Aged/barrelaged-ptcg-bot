# N22R Retry Execution Lock

Status: **frozen before count-corrected retry**

- governing tie-correction amendment SHA-256:
  `99afca5165703ad1b60882486cba67021a776a160025a5a70305535f5c4d59c7`;
- manifest-count correction SHA-256:
  `541c60f751e602c7b69c7af92499758990b4f15850b78d7fa315e522d0fa5d05`;
- corrected audit script SHA-256:
  `7ca495f3fc6f2ed3da4d394791fa2b317b276853fbd6a85df83002538bd00146`;
- quarantined N23 output SHA-256:
  `873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb`.

The prior N22R attempt failed before loading examples because of the documented
manifest-count literal. The output remains absent. This retry uses the same
sole command:

```powershell
python scripts/audit_n22r_first_index_ties.py
```

No other source, threshold, input, output, metric, or gate changed. Passage
authorizes only the one corrected held-out evaluator described by the governing
amendment.
