# September 2026 release privacy review

The V89/Ray/MinIO update removes private training paths from 28 ONNX files and
27 JSON reports in the current tree. All normalized model graphs are byte
identical, and three deterministic sampled inference comparisons per model
produced zero output differences. New download checksums are recorded alongside
the artifacts. Numeric evaluation results and original historical evidence
hashes are preserved.

A private exact-value audit compared five existing operator credentials
(submission API, Ray, MinIO, GitHub and model-storage credentials) against
7,676 reachable Git blobs across the lab and its three pinned submodules,
and against intended publication files. It found zero matches. Pattern scans
also checked API-token formats, private keys, literal passwords, authorization
links, private overlay addresses and machine paths. No credential findings were
identified; historical machine paths remain in earlier Git versions. The audit
never printed credential values and its private inputs are not committed.

MinIO support was verified by uploading and downloading a random 4,096-byte
artifact through the new helper, checking its SHA-256 and removing the temporary
object and bucket. No deployment addresses or passwords are included here.

This is a bounded audit of known credentials and detectable formats, not proof
against every arbitrary encoded secret. Git history, forks and external model
mirrors are not rewritten by this update.
