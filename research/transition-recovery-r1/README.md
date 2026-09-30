# Frozen-component transition recovery diagnostic

Archived research evidence. No combined model is promoted by this diagnostic.

Earlier GPU combined validation passed rotation error in all 48 lanes (under 10 degrees), with no body-contact/fall failures. Only 10 lanes passed every criterion; 18 failed final forward speed, with additional missing passes attributable to trick-tail forward speed or yaw settling. Those two omitted metrics must be measured explicitly; a transition-only cause is not yet established.

Hypothesis: an extended rolling recovery and gradual race engagement improve successful combined execution without changing either neural policy. Fixed roster: two reset seeds (98011 and 98012), 48 worlds each, three arms, 36 seconds per episode; 288 total simulated attempts. Arms: published trick then continued skating, direct race engagement, and eight-second recovery plus two-second blend into racing. Racing runs at 10 Hz, locomotion at 50 Hz, physics at 200 Hz. This is a new matched diagnostic, not an exact replay of the earlier 26-second test with 50 Hz route updates.

Reset-only joint noise is uniform +/-0.005 radians, identical by hash across arms. No state corrections during a trajectory. V89 and racing weights remain frozen. All arms preserve observation history. Record original trick-tail, recovery-tail, race-tail, and final skating speeds/yaw/heading plus cumulative body/fall status by phase.

Recovery commands forward 0.45 and heading correction clipped to +/-0.6 for eight seconds. Race engagement requires the final two recovery seconds to average >0.15 m/s forward speed, <0.5 rad/s absolute yaw rate, <20-degree maximum heading error, with no previous body contact/fall. Failed lanes do not engage racing and count as failed. Preserve original 15-second trick-tail results separately; longer recovery does not retroactively pass the published trick contract. A diagnostic pass requires all 96 recovery cases to meet recovery, race-engagement, no-contact/fall and final skating-speed checks. Subsequent course and standalone-skating/trick checks remain required; open-floor synthetic straight-route observations cannot establish arbitrary track rejoining.

One Ray GPU task with no retries and a two-hour limit per stage. The reserved simulator machine is excluded. No new training is launched by this diagnostic; retain every outcome without extending the roster to chase a pass.

## Preserved outcome

Extended recovery passed 36/48 and 40/48 checks (76/96 total), versus 8/48 and 2/48 (10/96) for direct engagement. The full 96/96 diagnostic gate failed. Race-tail speed and heading remain unqualified; there is no demonstrated track rejoining or promoted combined model. See `evaluation-summary.json`.

These exact worker sources require the prepared original environment and a privately supplied `racing-source.pt`. Raw optimizer checkpoints, job receipts and logs are excluded. Relative home-directory imports describe an operator environment template.
