# DuckWing V61 · V57b Control Fusion

V61 is the strongest qualified fusion produced from the September 1 RTX 5090
transfer. It combines the existing control-aware V11/Speed policy with the
transferred V57b speed actor through a command-aware ONNX router.

It is a **qualified frontier candidate**, not the all-around champion. It cuts
the official 100 ft time to 27.45 s and stays straighter than Pollen, but its
first-second acceleration is below both Pollen and the current control-aware
champion. The control-aware champion therefore remains the deployment default.

Official replay contract:

- wheel frictionloss: `0.003`
- current limit: `1.75 A`
- line hold: yaw Kp `0.55`, lateral Kp `0.10`, yaw Kd `0.0422`
- line hold maximum correction: `0.18 rad/s`
- speed route: V57b contribution `0.85`, tapered from yaw command `0.08` to `0.25 rad/s`

The ONNX actor retains the shared 61-observation / 14-action deployment
contract. Line-hold gains are controller configuration and are not embedded in
the ONNX graph.
