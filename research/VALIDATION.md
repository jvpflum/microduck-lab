# Publication checks — 2026-09-29

This update publishes research evidence; no model is newly promoted.

- Dashboard/receipt tests: 54 passed. Cleanup now resolves the checkpoint
  argument and checks actual containment, so sibling folders or unrelated
  arguments cannot select a process for cleanup. macOS temporary-directory
  aliases are handled in the test fixture.
- Home Ray contract tests: 5 passed, including explicit tested-simulator
  capability acceptance and rejection of an unverified capability.
- Preview-controller qualification tests: 5 passed.
- Archived Isaac adapter contract tests: 6 passed. These tests do not establish
  behavioral parity; the preserved simulation evidence still fails it.
- V90 adapter checks passed: unchanged V89 base, skating command identity,
  exact published trick commands, race mapping and explicit mode handoff.
- All 39 entries in the eight release SHA256SUMS files verified.
- Current-file, Git-history and additional binary/string privacy reviews
  returned zero known-pattern findings. ONNX metadata was parsed with ONNX
  installed; no new raw optimizer files or private receipts are published.
- Original MuJoCo-Warp Ray full-GPU learning smoke passed with actual CUDA
  physics, motor inference and PPO parameter updates. Long training and its
  held-out checkpoint comparisons remain pending.

The full fresh-clone Linux bootstrap/preflight/upstream test sequence was not
rerun for this publication. Local `make` targets are blocked by the macOS Xcode
license state, and this checkout's GPU simulator setup is Linux-specific.
The dashboard tests above ran directly through the repository shell script;
GPU readiness was verified in the existing prepared Linux worker via Ray.
No license was accepted or system GPU software changed to bypass this limit.
