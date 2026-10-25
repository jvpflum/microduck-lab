# Laguna Seca course navigation

Laguna is a separate higher-level control program using retained V89 roller
locomotion. It is not a new barefoot walking model.

The [V90 Laguna bundle](../../releases/v90-laguna/) contains the course controller,
source, videos and measured evidence. The flat robot-scale course excludes the
Corkscrew elevation. Held-out clean laps were 42/48 at each tested start-heading
offset; combined skate/trick/race transitions remained experimental (10/48).
V89 remains the recommended general-purpose roller model.

See the [research ledger](../../research/README.md) for rejected continuations and
[transition recovery](../../research/transition-recovery-r1/) for retained failures.
The historical [Isaac transfer diagnostic](../../research/isaaclab-v89-parity/)
is distinct from the current [barefoot workshop program](../isaac-workshop/).

The useful transferable method is hierarchical control: navigation generates
commands for a retained locomotor. Its privileged racing-state inputs and roller
weights do not establish camera-only barefoot navigation.
