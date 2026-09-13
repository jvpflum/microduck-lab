# DuckWing V89 Tricks 2

A composed inference release: the privacy-normalized, unchanged V89 ONNX
policy plus a reusable command-level trick controller. The tricks are not
encoded in new neural weights. Load both components to use them.

## Using a trick

Continue normal V89 inference and its continuous observation history. At the
start of a trick, create a `TrickController` and call
`start(heading_radians, direction=1, degrees=360)`. Use direction -1 for the
opposite turn. On each 50 Hz control tick, call
`update(heading_radians, ordinary_13_command_vector)` and feed the returned
commands through the ordinary observation builder into V89. The returned
commands override forward, lateral and yaw commands while active. All joint
actions still come from V89. `cancel()` immediately returns command ownership
to the caller. Inactive operation passes the original commands through.

The controller keeps a bounded fifteen-second turn/alignment window, then
returns ownership to the caller. Its `heading_complete` status only checks
heading; it is not a substitute for the contact, tilt and recovery evaluator.
Do not reset observation history or physics at a trick boundary.

## Validated setup and limits

Trials start at rest, idle two seconds, then approach for three seconds using
forward command 0.30. Tricks use a 0.60 yaw-command limit and heading feedback.
The control physics is official CPU MuJoCo at 200 Hz and 50 Hz control,
wheel frictionloss 0.003 and actuator torque limit equivalent to 1.75 A.
This release does not use the GPU BAM flip dynamics.

Qualification requires requested signed rotation within 15 degrees, no body
ground contact, maximum tilt below 45 degrees, final-two-second heading error
below 20 degrees, mean absolute yaw rate below 0.5 rad/s and rolling speed
above 0.15 m/s. See validation.json for each measured trial.

Simulation yaw is the heading source in these tests. Hardware use would
require validated heading estimation and independent physical testing.
Only nominal friction and small initial joint perturbations were tested.
The release has not been wired into the app's trick buttons or deployed to a
physical robot. The original V89 release and its weights remain unchanged.

`SHA256SUMS` covers every bundled file other than itself. `manifest.json`
identifies the included tricks and validation status. `base-provenance.json`
records why the published ONNX hash differs from the private evaluator model:
only metadata containing private paths was removed; its graph is identical.

## Double spin

Use degrees=720 for two consecutive skating rotations. Fresh Ray validation
passed 16/16 trials, both directions. Time to the 705-degree threshold was
2.86–4.40 seconds, followed by heading alignment and controlled rolling exit.
The final two-second heading tolerance is 20 degrees relative to 720 degrees.
See validation-720.json and skate-720.mp4. The original 180/360 controller
behavior is unchanged; only acceptance of the new 720-degree target is added.
