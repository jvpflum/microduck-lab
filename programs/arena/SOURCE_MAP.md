# Arena source and evidence map

Prepared 2026-10-25. This is a selected, dated inventory of inspected campaign
evidence, not an exhaustive archive or current job/leaderboard status.

| Public artifact | Source and transformation |
| --- | --- |
| `fresh-sprint/env.py` | Exact historical fresh roller PPO environment. BAM batching attribution remains in the source; no physics or reward edits for publication. |
| `fresh-sprint/train.py` | Historical fresh PPO trainer. Only the owned-checkpoint NumPy RNG representation and safe reload changed; no external model import or launch service added. |
| `fresh-sprint/evaluate_entry.py` | Exact retained CPU Arena evaluator wrapper. Requires the pinned official source tree and complete hashes supplied separately. |
| `fresh-sprint/test_env.py`, `test_train.py` | Original interface/numerical tests, copied unchanged. |
| `fresh-sprint/checkpoint_state.py`, `test_checkpoint_state.py` | New portable primitive RNG codec and rejection/round-trip guards for `weights_only=True`. |
| `evidence-inventory.json` | Selected numeric fields and hashes from retained completion, selection, parent-binding, failed-continuation and upright-smoke receipts. No raw operator record is included. |
| `fresh-sprint/source-manifest.json` | Original and public source hashes, exact serialization-adaptation scope. |
| `validation.json` | Publication checks and explicitly unavailable native numerical tests. |

The source receipts remain private because they contain machine and operator
metadata. Their byte hashes establish provenance, not public access to the raw
traces. These summaries are not a self-contained independent replay.
The evaluator source and pinned official revision provide the recipe for a new
attributable run with separately supplied assets and candidate.

The failed continuation's 2500 checkpoint never reached race judging: its
normalized actor/ONNX difference was 2.288818359375e-5 against the unchanged 2e-5
gate. It remains failed. The 2.680-second parent and 2.864-second intermediate
have distinct evaluated checkpoints; PPO reward and extra training steps are not
substituted for an evaluation result.

The 1.648-second report is unverified within this publication. The event-family
inventory establishes the reviewed event definitions only. No aggregate Olympic
result, live submission, new release, physical-robot qualification or generalized
walking capability is asserted.
