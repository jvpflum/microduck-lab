# Six MicroDucks in one physical attic

**The two-second native six-robot smoke passed independent audit; the full
20-second recording and movie remain pending.** Earlier native startup failures
are retained. This short shared-world episode does not establish robust walking,
autonomous behavior or locomotion qualification. The existing
`media/walk-turn-stop.mp4` is the earlier single-robot experiment.

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
The full recording has been queued separately. The curated numeric evidence
and source-audit hash are in [evidence-summary.json](evidence-summary.json).

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
- Then record a separate continuous 20-second episode: 1,000 controls at 50 Hz
  and 4,000 solved substeps at 200 Hz. Command forward at 0.15 m/s for eight
  seconds, stop for four, walk for four, then stop for four.
- Retain every actor's raw observation, command and policy output, actual joint
  state, all solved-body poses and raw contacts. Verify foot contact per robot
  and preserve any inter-robot contacts.
- Stop before autoreset on a fall, nonfinite state or domain escape. A partial
  episode remains partial; resets cannot be spliced into a complete movie.
- Require independent source, model, timeline, contact and CPU policy-replay
  checks before admitting the full episode to rendering.

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

The [proposed shot plan](film-shot-plan.json) selects five contiguous views of
that same episode. At 25 fps it samples controls 0, 2, …, 998 for 500 frames;
the final control-1000 state is retained as evidence beyond the last displayed
sample. Camera motion follows measured robot roots without altering actor
poses. Actual framing, lighting, capture integrity and achieved rendering
quality still need native verification. Requested path-tracing settings are
not measured sample counts.

## Portable source boundary

`film_motion.py` publishes the exact command schedule, per-actor action-history
check and link-pose matrix conversion, plus a guarded stage-unit wrapper.
`film_cameras.py` publishes the authored camera plan validator, measured-root
camera sampler and session-only USD camera authoring helper. USD imports are
lazy. Importing the module and running the CPU tests does not require Kit or
prove native compatibility.

The [source map](SOURCE_MAP.json) binds the original authored components and
public copies by hash. Tests use synthetic poses and fake action managers;
they verify unit conventions, role separation, camera coverage and rejection
of malformed input. They cannot establish real contacts or policy quality.

This bundle excludes the native scene/robot assets, checkpoint tensors,
private orchestration and admissions, raw poses, per-job logs and generated
film. Obtain licensed source assets separately. The existing public scene
layers remain unchanged. No DuckWing release, Arena policy or protected
interface is changed by this work.
