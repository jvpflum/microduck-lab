# DuckWing V67 · Joint-Specialist Fusion

V67 is the new simulation-qualified DuckWing leader. It keeps V66 as the
incumbent control policy, imports propulsion from the V47 official-friction
speed specialist, and routes moving zero-command states to V65 for stable
braking. The high-speed fusion uses 25% V47 authority on the hip yaw/roll
steering joints, 105% authority on the hip pitch/knee/ankle propulsion joints,
and no V47 authority on the head/neck joints.

On two identical deterministic Spark replays, V67 improved every primary V66
Race5 measurement and passed the independent high-speed-braking and idle gates.
It passes all 15 Policy Bench qualification gates and retains all 9 comparable
wins over the official Pollen roller.

Official replay contract:

- wheel frictionloss: `0.003`
- current limit: `1.75 A`
- line hold: yaw Kp `0.70`, lateral Kp `0.22`, yaw Kd `0.07`
- line hold maximum correction: `0.15 rad/s`
- speed route: active for command X above `0.5 m/s`, with specialist authority
  tapered between yaw commands `0.08` and `0.25 rad/s`
- moving brake route: V65 while command X is effectively zero and mean absolute
  joint velocity exceeds `0.20 rad/s`

The ONNX policy retains the shared 61-observation / 14-action deployment
contract. Line-hold gains are controller configuration and are not embedded in
the ONNX graph. Hardware validation is still required.

The 5 MPH goal has not been reached: V67 delivers 2.240 MPH sustained in the
eight-second race phase, 2.727 MPH sustained in the long 100-foot phase, and a
3.060 MPH verified top speed under the official physics contract.

The subsequent V68 local-evolution search did not displace V67. Across 586
scored joint-authority and line-controller configurations, no challenger
improved speed while preserving every V67 stability, tracking, acceleration,
contact, and braking measurement. See [V68-SEARCH.md](V68-SEARCH.md) for the
best near-miss and the next training strategy.

V69 then tested 150 state-guarded blends that gave the fastest V68 challenger
authority only at low body-yaw rates and returned exactly to V67 above a yaw
threshold. Nine variants improved speed, but every one lost at least one
required stability or control measurement; the fastest was 0.95% quicker over
100 feet but increased heading error by 20.2%. V67 therefore remains the
strict, lossless leader. See [V69-SEARCH.md](V69-SEARCH.md) for the complete
result and the recommended next experiment.

V70 implemented that recommendation as a deployable teacher-guided residual
run with contact/stride shaping. It completed cleanly but plateaued below V67;
its conservative compositions improved drift or heading only by giving back
speed, tilt, or contact. V67 remains definitive. See
[V70-SEARCH.md](V70-SEARCH.md) for the checkpoint table and next strategy.
