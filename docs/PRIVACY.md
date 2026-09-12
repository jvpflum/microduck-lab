# Private infrastructure and public releases

Public Git contains portable code, inference actors, checksums and reviewed
numeric evidence. Machine addresses, usernames, absolute training paths,
Tailscale authorization links, service environment files, credentials, private
keys, raw job receipts and optimizer state must stay outside the repository.
Use operator environment variables or mode-0600 ignored secret files. Never
put API tokens or MinIO passwords in command examples, URLs, logs or screenshots.
Keep Ray's authentication secret separate from the submission API token.
MinIO accounts should have private-bucket permissions scoped to their workload.

The metadata privacy cleanup changes only ONNX descriptive metadata; graph
bytes and sampled inference outputs are identical. Download checksums and model
manifests reflect the sanitized files. Historical evaluator hashes remain as
provenance for the original evaluated exports. See `privacy-normalization.json`.
The private training copies and released V80 research contract are unchanged.
New releases must normally remain immutable; this explicitly recorded privacy
cleanup is an exception for removing machine paths from public downloads.

Removing a file or metadata from the current branch does not erase earlier
Git commits, forks, caches or external downloads. If an actual credential is
ever found in history, revoke or rotate it first; history rewriting alone
cannot make an exposed credential safe. Infrastructure addresses and paths
already in history also remain discoverable. Do not claim a scan proves that
no arbitrary encoded secret exists; review scanner findings and binary metadata.

Before publishing, run `python3 tools/audit_public_privacy.py`, inspect the
staged diff, verify release checksums and run applicable tests. The scanner
prints filenames and categories, never matching secret values. It scans tracked
files and intended additions, including ONNX metadata when ONNX is installed.
