# DuckWing V66 · V65 Control Fusion

V66 is the all-around DuckWing champion produced from the September 1 RTX 5090
V65 transfer. It routes the existing control-aware champion at idle, cruise,
and turning commands, then blends in 96.5% of V65 for straight high-speed
commands. V65 itself embeds V63 for mid-speed control and V59 for high speed.

On Spark's canonical official-friction replay, V66 passes all 15 control gates
and beats Pollen on all 9 leaderboard dimensions. It improves the previous
all-around champion's 100 ft time from 29.47 s to 26.32 s while also improving
sustained speed and long-run drift.

Official replay contract:

- wheel frictionloss: `0.003`
- current limit: `1.75 A`
- line hold: yaw Kp `0.70`, lateral Kp `0.14`, yaw Kd `0.07`
- line hold maximum correction: `0.15 rad/s`
- speed route: V65 contribution `0.965`, tapered from yaw command `0.08` to `0.25 rad/s`

The ONNX actor retains the shared 61-observation / 14-action deployment
contract. Line-hold gains are controller configuration and are not embedded in
the ONNX graph.
