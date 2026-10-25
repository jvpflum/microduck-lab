# Authored workshop scene layers

This is the actual authored OpenUSD layer stack for the custom toy-inventor attic,
including perimeter storage, central colorful duck crew, workbench, lighting,
cameras, dock, material bindings and paint station. The cabinet-clearance override
is retained separately under `cabinet-clearance/` and must be composed above the
main stack in a provisioned runtime. `SOURCE_MAP.json` binds original and public
file hashes. One descriptive operator-path metadata string was replaced by an
external-content placeholder; no geometry was changed.

The original NVIDIA Sample Scenes / Old Attic meshes, textures, MDL and vMaterials
pack are **not redistributed**. Native MicroDuck robot source USD is also external.
The relative references retain the original acquisition layout; cloning this
repository alone will not resolve those dependencies or render the complete scene.
Use [NVIDIA Sample Scenes](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
and the pinned Isaac Lab robot source through their own distribution and licensing
terms, preserving their relative dependency paths. The active runtime uses Isaac
Lab revision `b03345bbf2c9a54eb03f9bd823ae19c0ca3a014e`.

These are render-inspection layers. Keep the timeline stopped for that profile;
inherited disabled-body flags are not a guarantee that Newton skips importing
those bodies. Neither source parsing nor the preview establishes dynamics,
breakage, navigation or physical-robot safety. Full-body training uses a separately
validated bounded collision environment, not all decorative objects as solved
colliders. The bench duck and parked companions are visual-only.

The program preview shows an actual native render. The retained capture-log failure
and current locomotion/camera limitations remain documented in the program status.
