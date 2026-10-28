# Swappable workshop themes

The seasonal overlay keeps the attic, robot geometry and recorded motion unchanged.
Halloween adds four imported painted-toy props, supported bats, a cloth ghost,
a web, eleven warm practical bulbs and reversible overrides on six named lights.
Neutral removes the seasonal layer. Christmas is a **starter** with a red/green
ornament garland; it has no acquired Christmas prop pack, soundtrack or accepted
native preview.

On 2026-10-28, five actual Halloween views passed all **13 capture-integrity
checks** and parent-agent image inspection. The complete 300-frame, 12-second film remains
pending. These are render checks, not locomotion qualification. The recorded
source still includes the later fall and the selected excerpt's known self-contact
and nearly stationary duck; see [the film account](SIX_ROBOT_FILM.md).

## Portable authoring

The public generator uses the reviewed placement and decorative geometry formulas,
with explicit input/output folders instead of an operator's project layout.
It requires Python's standard library. It does not install, fetch, simulate or
render anything. Vendor scenes, robot meshes, converted props, textures, recordings
and audio are **not included**.

1. Copy `seasonal-profiles/` into a configuration folder outside the checkout.
2. Copy `anchor-contract.example.json` to `anchor-contract.json`. Supply your
   base scene and its complete dependency closure in a separate source folder.
   Replace the intentionally invalid hash placeholder, enumerate every dependency
   in `source_pins`, and remeasure anchors, blocked bounds and robot exclusions for
   that exact composition. The coordinates shown are an attic example, not safe
   universal placement coordinates.
3. For Halloween, supply a verified external asset catalog at
   `assets/kaykit-halloween/assets.json` relative to your asset folder. See
   [the catalog interface and provenance](seasonal-profiles/ASSETS.md). Neutral
   and Christmas have empty catalog lists and need no Halloween assets.
4. Build into a fresh output folder, then select it. From this program directory:

```sh
PYTHONPATH=src python3 -m isaac_workshop.seasonal_theme halloween \
  --config-root ./inputs/config --source-root ./inputs/scene \
  --asset-root ./inputs/assets --output ./build/halloween-v1 \
  --selection-root ./selected
```

Use a new output directory for each build. Repeat with `christmas` or `neutral`
and a fresh output while keeping `--selection-root ./selected`. The selection
contains exactly the chosen theme and base; it does not accumulate older overlays.
Close the old light clock and reopen `selected.usda` in the consuming application.
The ownership marker prevents replacing an unrelated selection directory.

The contract uses Z-up centimetres and requires a measured identity transform on composed `/Root`. Authored anchors are `/Root`-local; the world-bound checks are valid only when those frames coincide. Verify inherited transforms in the complete composition and declare `root_transform_identity: true` only after that check. The stdlib builder validates this declaration, not composed USD transforms. Assets use Z-up metres with `/Asset` as their
root, no physics, finite inspected bounds and a complete hashed dependency list.
The builder seats each asset's actual declared minimum Z, centers its XY footprint,
applies target height and yaw, and checks the conservative footprint against its
measured support rectangle, existing-object bounds and protected robot volume.
The decorative frame recipe assumes posts at local X ±59 cm, Y 0, reaching Z
100–168 cm. Its anchors must describe those real supports. Hash checks protect
**declared** files; this small stdlib tool is not an automatic USD dependency
resolver or a geometry/collision validator. Verify composition and actual support
in OpenUSD before using a different scene. It authors no robot or physics API.

Only explicitly listed light paths and value attributes can override base lights.
The example includes both legacy source fields and canonical `inputs:*` fields;
actual canonical readback and native appearance must be checked after schema
initialization. Removing the theme restores the base rather than overwriting its
defaults. Imported candles and pumpkins are diffuse toys; separate visible bulbs
provide the authored practical lighting.

## Render integration

`seasonal_runtime.ThemeClock` owns one anonymous session sublayer. Call
`apply_theme_time(clock, frame / fps)` explicitly before each capture and `close()`
on every exit/switch. This modulates only the eleven seasonal intensity values;
it does not advance the simulation or change robot poses. A stopped timeline does
not automatically play light animation. `canonical_light_readback` verifies
values and records types, but does not prove their visible effect. A consumer must
also enforce expected types, frozen poses, camera/capture provenance and clean
native logs. No native submitter or runtime admission is published here.

Profiles have independent title/audio slots. All public audio slots are empty;
provide your own rights-checked soundtrack and credits separately. Switching to
Christmas or neutral must also replace the consumer's media selection.

The separate [seasonal camera plan](film-seasonal-shot-plan.json) and
`film_seasonal_cameras.py` preserve measured XY motion, use fixed camera height for
two close shots, and include a wider fixed room view. They never smooth or relocate
a robot. They are specific to the original centimetre stage and six named roots.
The older 20-second and excerpt camera modules/plans remain unchanged.

## Validation scope

Synthetic CPU tests exercise source/asset tamper rejection, path containment,
seating and bounds, tether connectivity, neutral independence, persistent
Halloween → Christmas → neutral replacement, disposable light-layer ownership
and deterministic light sampling. They require no licensed source assets.
They do not reproduce the full native scene or establish physical safety,
robust walking, autonomous navigation, or final film completion.
