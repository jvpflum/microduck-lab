# Xbox controller for mjlab/Viser

The Viser launcher includes a local browser Gamepad bridge. The browser reads
the controller attached to the operator's computer and sends bounded commands
to the Spark over an SSH-forwarded HTTP connection. No USB or Bluetooth device
is forwarded to the Spark.

## Connect

On the operator computer:

```bash
ssh -L 8080:localhost:8080 -L 8090:localhost:8090 <ssh-user>@<spark-address>
```

In that SSH session, launch the original roller policy:

```bash
~/projects/microduck-lab/scripts/view-final-skate.sh
```

Open both pages locally:

- Viser: <http://localhost:8080>
- Controller: <http://localhost:8090>

Connect the Xbox controller to the operator computer, press a controller button
so the browser discovers it, and click **Arm controller**. The controller bridge
overrides Viser's generated command only while armed.

## Mapping

- left stick Y or triggers: signed propulsion;
- right stick X: heading error;
- `X`: coast/zero command while held;
- `A`: reset the environment;
- `Start`: pause/play;
- `B`: latch emergency zero.

The page has separate semantics presets:

- **Original roller:** negative propulsion means brake, and heading should remain
  zero because that checkpoint was trained straight-only.
- **Swizzle:** negative propulsion means reverse and heading is trained to ±0.5.
  Active braking is a future separate command/policy.

## Safety behavior

Commands are clamped server-side. If the browser, controller, or network stops
updating for 500 ms while armed, propulsion and heading are forced to zero. A
disconnect also forces zero. Emergency zero is latched in the browser until the
operator clears it.

This is a simulation control surface, not a physical-robot safety system. A real
deployment requires a hardware emergency stop, an independent runtime watchdog,
conservative limits, and authenticated command transport.
