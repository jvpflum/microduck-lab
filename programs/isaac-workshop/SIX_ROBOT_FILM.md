# Six MicroDucks in one physical attic

**The two-second smoke passed; the later 20-second attempt failed when
`duck_1` fell at 15.16 seconds.** A continuous 12-second prefix has passed
source-evidence review for visualization. The movie remains pending.
Neither attempt establishes robust walking, autonomous behavior or locomotion
qualification. Earlier failures remain retained. The existing
`media/walk-turn-stop.mp4` is the earlier single-robot experiment.

## Latest full-attempt outcome and selected excerpt

The requested 1,000-control episode stopped naturally at control 758, after
3,032 solved physics substeps. The fall was recorded before autoreset, with
its action, all four final substeps and final body states intact. No reset
occurred. The requested full horizon **failed**; a successful evidence audit
does not change that outcome.

The renderer may use only the uninterrupted original controls 0–600: eight
seconds of commanded walking followed by four seconds of commanded stopping.
It validates the entire retained 758-control source before selecting the 601
prefix states. No loops, retiming, pose blends, splices or invented movement
are allowed. The later fall remains disclosed even though it is outside this
selected interval.

| Robot | Planar endpoint displacement in selected 12 seconds |
| --- | ---: |
| `duck_0` | 0.50784 m |
| `duck_1` | 0.00436 m |
| `duck_2` | 0.49802 m |
| `duck_3` | 0.25824 m |
| `duck_4` | 0.55560 m |
| `duck_5` | 0.59535 m |

`duck_1` remained nearly stationary: its maximum planar excursion from its
initial root position was 0.01112 m across all 601 prefix boundaries. These
endpoint displacements are not total travel distances. `duck_3` had measured
leg/battery self-contact inside the excerpt, first observed at 1.395 seconds.
Its accumulated nonsole force-magnitude measure was 0.971893475 N·s. This sums
per-shape force magnitudes, including both signed body views of self-contact;
it is neither net external impulse nor a unique-pair impulse.

The full source audit retained 4,548 actual actor outputs; CPU replay differed
from native output by at most 9.536743 × 10⁻⁷. No inter-duck contact occurred.
The audit proves retained evidence consistency, not successful walking,
stopping, collision response or autonomy. See the curated metrics and exact
audit hashes in [evidence-summary.json](evidence-summary.json).

## Actual two-second result

Six physical robots completed 100 controls and 400 solved physics substeps in
one world. All six made measured sole-to-floor contacts. No nonsole force,
falls or autoresets were observed. The independent audit replayed all 600
actual actor outputs on CPU; maximum absolute difference from native output
was **2.682209 × 10⁻⁷**. This checks inference consistency, not skill.

| Robot | Planar displacement over two seconds |
| --- | ---: |
| `duck_0` | 0.07965 m |
| `duck_1` | 0.01068 m |
| `duck_2` | 0.11518 m |
| `duck_3` | 0.05521 m |
| `duck_4` | 0.13019 m |
| `duck_5` | 0.07623 m |

Progress was small and uneven. No inter-duck contact occurred, so this episode
does not test inter-duck collision response. The smoke covers only the first
walking segment; stopping and the complete 20-second schedule remain unverified.
This was a separate earlier run; its zero-nonsole-contact result does not apply
to the later full attempt. Its numeric evidence and source-audit hash remain
in [evidence-summary.json](evidence-summary.json).

The prototype places six separately controlled MicroDucks in **one physics
world**. Each uses the same retained native walking foundation, with its own
61-value actor observation, 76-value critic observation and 14-action history.
Named routing prevents one robot from reading another robot's state or the
combined 84-action history. No policy learning occurs during film capture.

The verified smoke binding contains 174 link bodies, 168 hinges, 84 controlled
motor actions and 60 robot collision shapes. Twelve authored within-robot
collision exclusions are retained; inter-robot contacts have no exemptions.
The same binding must be reverified for the full recording. Enabling collision
pairs does not prove a collision response was observed.

The bench maintenance duck remains a static display. The six floor-display
copies are replaced by the physical actors for the episode. The admitted
walking corridor uses two source-derived static floor meshes; room decoration
outside that collision domain is not claimed to be physically interactive.

## Bounded episode and acceptance

- Completed: independently verify 100 controls / 400 solved substeps as a
  two-second smoke.
- Original request, not achieved: a separate continuous 20-second episode,
  1,000 controls at 50 Hz
  and 4,000 solved substeps at 200 Hz. Command forward at 0.15 m/s for eight
  seconds, stop for four, walk for four, then stop for four.
- Retain every actor's raw observation, command and policy output, actual joint
  state, all solved-body poses and raw contacts. Verify foot contact per robot
  and preserve any inter-robot contacts.
- Stop before autoreset on a fall, nonfinite state or domain escape. A partial
  episode remains partial; resets cannot be spliced into a complete movie.
- Require independent source, model, timeline, contact and CPU policy-replay
  checks before admitting any recorded interval to rendering. Full-episode
  admission remains separate from the explicit partial-source excerpt.

The timed walk/stop command is a showcase protocol, not navigation, obstacle
avoidance, coordination or a training curriculum. The walking foundation is
unqualified for robust turning and unseen routes.

## Render the measured motion

The renderer is designed to replay the exact recorded native link-origin poses
in a stopped USD timeline. It does not use center-of-mass poses, forward
kinematics, pose interpolation, artificial root travel or independent-world
compositing. All 174 body poses must map to verified source mesh/link frames.
When converting metre-local robot meshes into the centimetre scene, both the
transform basis and translation must be scaled; the current parent transform
must then be inverted. Otherwise the robot can shrink or move twice.

The [original 20-second shot plan](film-shot-plan.json) and its helpers remain
unchanged as the record of the original request. It cannot be fulfilled from
the partial recording. The separate [12-second excerpt plan](film-excerpt-shot-plan.json)
selects five contiguous 60-frame views. At 25 fps it samples controls
0, 2, …, 598 for 300 frames; state 600 at 12 seconds is validated beyond the
last displayed sample at 11.96 seconds. Camera motion follows measured robot
roots without altering actor poses. Actual framing, lighting, capture integrity
and achieved rendering quality still need native verification. Requested
path-tracing settings are not measured sample counts.

## Portable source boundary

`film_motion.py` publishes the exact command schedule, per-actor action-history
check and link-pose matrix conversion, plus a guarded stage-unit wrapper.
`film_cameras.py` publishes the authored camera plan validator, measured-root
camera sampler and session-only USD camera authoring helper for the original
20-second plan. `film_excerpt_cameras.py` is the separate 12-second version;
neither replaces the other. USD imports are lazy. Importing the modules and
running the CPU tests does not require Kit or prove native compatibility.

The [source map](SOURCE_MAP.json) binds the original authored components and
public copies by hash. Tests use synthetic poses and fake action managers;
they verify unit conventions, role separation, camera coverage and rejection
of malformed input. They cannot establish real contacts or policy quality.

This bundle excludes the native scene/robot assets, checkpoint tensors,
private orchestration and admissions, raw poses, per-job logs and generated
film. Obtain licensed source assets separately. The existing public scene
layers remain unchanged. No DuckWing release, Arena policy or protected
interface is changed by this work.
