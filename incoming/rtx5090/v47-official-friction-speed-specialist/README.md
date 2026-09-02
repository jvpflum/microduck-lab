# V47 official-friction speed specialist

Curated deployable donor used by DuckWing V67. This is a straight high-speed
specialist, not a complete drive controller; V67 supplies command routing,
turning, idle hold, and braking.

- wheel frictionloss: `0.003`
- evaluator current limit: `1.75 A`
- policy SHA-256: `6079db680499a771ef34a9d391b97eee4276332df54bd3a461a7362e021add87`
- native specialist result: 100 ft in 26.145 s, approximately 1.204 m/s steady
  speed, 0.521 m maximum drift, and 18.2° maximum heading error

Only the deployment ONNX is retained; no raw checkpoint, optimizer state,
training log, or credential is included.
