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
overrides Viser's generated command only while armed. No Viser checkbox or
Twist toggle is required.

The page prefers controllers that expose the browser's standard gamepad
mapping, but it no longer assumes the first connected device or one fixed axis
layout. Use the **Controller** menu when more than one device is present. If an
unmapped controller does not respond automatically, move its stick and watch
**Raw axes**, then choose the changing axis under **Propulsion source** or
**Heading source**. The choice is saved in the browser.

For an end-to-end check, arm the page and hold **Hold to test forward**, or use
`W`/`S`. The **Sending** line must become nonzero and MicroDuck should respond
without controller hardware. This distinguishes a browser mapping problem from
a simulator bridge problem.

## Mapping

- left stick Y or triggers: signed propulsion;
- right stick X: heading error;
- `X`: coast/zero command while held;
- on-screen **Reset robot**: reset the environment;
- on-screen **Pause / play**: toggle simulation playback;
- on-screen **Emergency zero**: latch propulsion and heading at zero;
- on-screen **Resume controls**: clear the emergency-zero latch.

The page has separate semantics presets:

- **Original roller:** negative propulsion means brake, and heading should remain
  zero because that checkpoint was trained straight-only.
- **Swizzle:** negative propulsion means reverse and heading is trained to ±0.5.
  Active braking is a future separate command/policy.

## Safety behavior

Commands are clamped server-side. If the browser, controller, or network stops
updating for 500 ms while armed, propulsion and heading are forced to zero. A
disconnect also forces zero. Emergency zero is latched in the browser until the
operator clicks **Resume controls**. It is deliberately not mapped to an Xbox
face button, where an accidental press can silently disable driving.
A new nonzero propulsion or heading command safely resumes the viewer if it was
paused; it does not toggle a viewer that is already running.

This is a simulation control surface, not a physical-robot safety system. A real
deployment requires a hardware emergency stop, an independent runtime watchdog,
conservative limits, and authenticated command transport.

## Policy Bench sessions

When a model is opened from Policy Bench, do not assume it uses 8080/8090. The
dashboard assigns a dedicated pair and lists it under **Open simulations**.
Always open the arena and Xbox controller from the same session card. This is
what keeps controller commands attached to the selected policy when multiple
models are being compared side by side.
