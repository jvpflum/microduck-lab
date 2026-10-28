# External seasonal asset interface

No source or converted assets are stored in this directory. The profile references
four external USD props from [KayKit Halloween Bits](https://github.com/KayKit-Game-Assets/KayKit-Halloween-Bits-1.0)
at commit `6dc69bf6b2fa766a985754f35ec6a0324090e6c6`:
`pumpkin_orange`, `pumpkin_orange_jackolantern`, `candle_triple`, `skull_candle`.
The pinned repository license is CC0-1.0. Exact source URLs, bytes, checksums,
conversion description and derived bounds are in [asset-provenance.json](asset-provenance.json).

Supply assets independently under the explicit `--asset-root`. The catalog path
is relative to that root; every asset/dependency path is also relative to the same
root, not to the catalog. Follow [asset-catalog.schema.json](asset-catalog.schema.json).
Duplicate IDs, escaped paths, changed hashes, wrong units/root/axis, or declared
physics are rejected. The generator checks metadata and bytes; it does not infer
whether a supplied USD contains undeclared dependencies or APIs.

For the original import, float32 glTF positions/normals were rotated `(x,-z,y)`
from Y-up to Z-up without changing size, UV T was flipped, and topology and the
single PBR atlas were retained. Source roughness was approximately 0.6 and
metallic was zero. The complete converter is not part of this small publication;
use a verified conversion and record its inspected bounds and complete closure.
Do not assume the pumpkins or candle atlas emit light.

Neutral and the Christmas starter list no catalogs. Adding a later Christmas
pack requires a separate catalog, provenance/rights review, measured supports and
native preview; it does not require hard-coding another package in Python.
NVIDIA scene/robot/material packs and audio/model weights are not redistributed.
