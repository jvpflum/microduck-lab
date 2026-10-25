# Fresh roller-sprint source snapshot

This is the portable source from the historical fresh PPO campaign, with one
explicit checkpoint-serialization adaptation. It is not a promoted policy or a
newly reproduced training result. No model weights or operator records are
included. Original and public source hashes are in
[source-manifest.json](source-manifest.json).

`env.py` owns the Arena-specific roller reset, observations, reward terms and
BAM chronology. `train.py` trains a fresh actor and critic; `evaluate_entry.py`
wraps the pinned official CPU evaluator without changing its physics, sweep or
failure scoring. The environment's BAM batching follows the attributed Apache-2.0
upstream implementation; see the source header and repository license.

The actor is 61→512→256→128→14 with an empirical normalizer. Its actual previous
actions occupy slots 34:48; forward command is slot 48. Slots **49:61 are forced
to zero**, including lateral and yaw commands. The critic has 69 inputs. This is
a forward roller race policy, not a general command-conditioned walking model.
Exports include the exact current normalizer and are checked against the actor
at an absolute tolerance of 2e-5. No tolerance was relaxed for this publication.

The historical runtime is Python 3.12, mjlab 1.3.0, MuJoCo 3.10.0,
MuJoCo Warp 3.8.1, Warp 1.12.0, RSL-RL 5.0.1 and Torch 2.9.1. The trainer enforces
these versions and CUDA learning; it does not install dependencies. The upstream
source revision recorded by the recipe is `a3e0c0fbe6c4574fa8b6fb203cd81f363c75e9be`.
A compatible upstream MicroDuck environment and source assets must already exist.

Run local contracts from this directory:

```sh
python3 -m unittest test_env test_train test_checkpoint_state
python3 train.py --help
python3 evaluate_entry.py --help
```

CPU interface tests need NumPy. Four numerical actor/export tests additionally
need the pinned Torch/RSL and ONNX runtime; they skip when Torch/RSL is absent.
They do not constitute native physics or GPU validation.

An explicitly chosen native smoke run uses:

```sh
python3 train.py --phase smoke --output ./new-smoke
```

That invocation **does perform training**. It does not reserve a GPU, submit a
competition entry or judge the policy unless both `--judge-contract` and
`--judge-python` are supplied. Historical bounds are 64 worlds × 5 updates for
smoke and 4,096 worlds × 3,000 updates for the train phase. A bounded phase's
completion is not promotion. Supply a new output directory each time.

For authoritative local judging, prepare a contract with the exact official
source revision, hashes for every runtime Python/asset file, the candidate ONNX
path and SHA, and the pinned CPU runtime. All paths must stay under the contract
root. Then run the evaluator in that runtime:

```sh
python3 evaluate_entry.py --contract ./judge-contract.json \
  --candidate chosen-policy --output ./new-judge
```

No incomplete example contract is supplied: missing source coverage is rejected.
The judge retains the golden check, selected speed, every start and failures.
Keep generated raw checkpoints, logs and local paths out of public Git.

The public trainer serializes NumPy RNG state as validated primitive values and
uses `torch.load(weights_only=True)` only to verify its own newly saved
checkpoint. There is no external checkpoint CLI or unsafe fallback. Local tests
prove RNG round-trip preservation and reject malformed state. This packaging
change has not been rerun in the historical native GPU runtime; a compatible
native smoke remains required before using it for new training. The environment,
PPO updates, reward formulas, normalizer and export threshold are unchanged.
