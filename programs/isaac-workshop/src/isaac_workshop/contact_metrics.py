"""Raw solved-contact metrics; no force threshold or self-contact exemption.

Force magnitude integrated over time is not net impulse or a damage measure.
"""
import math


def contact_metrics(substeps):
    """Measure exactly one four-substep control; no force threshold changes."""
    if len(substeps) != 4:
        raise ValueError('Exactly four solved physics substeps are required')
    control = substeps[0]['control_step']
    first = substeps[0]['physics_step']
    pairs = {}; active_rows = 0; total = 0.; peak = 0.
    for i, sample in enumerate(substeps):
        if (sample['control_step'] != control or sample['substep'] != i + 1
                or sample['physics_step'] != first + i
                or sample['boundary'] != 'after_native_solver_before_control_autoreset'
                or not math.isclose(sample['physics_dt_s'], .005, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Noncontiguous solved physics sample')
        evidence = sample['contacts']
        if not (evidence.get('all_finite') is True and evidence.get('all_worlds_observed') is True
                and evidence.get('contact_substeps_covered') == 1):
            raise ValueError('Contact coverage or finite evidence is absent')
        if evidence['native_contact_count'] != len(evidence['raw_rows']):
            raise ValueError('Raw contact rows do not match native count')
        seen = set()
        for row in evidence['raw_rows']:
            if row['row'] in seen or row['world'] != 0:
                raise ValueError('Duplicate row or unsupported world')
            seen.add(row['row'])
            f = row['force_on_shape0_world_n']
            if len(f) != 3 or not all(math.isfinite(float(x)) for x in f):
                raise ValueError('Invalid contact force')
            contacts = row['robot_contacts']
            if not contacts:
                continue
            illegal = any(c['robot_part'] not in ('sole_left', 'sole_right')
                or c['object_id'] != 'floor' for c in contacts)
            if illegal:
                magnitude = math.sqrt(sum(float(x)**2 for x in f))
                impulse = magnitude * .005
                total += impulse; peak = max(peak, magnitude); active_rows += magnitude > 0
                key = '|'.join(sorted(row['shape_paths']))
                pairs[key] = pairs.get(key, 0.) + impulse
    return {'nonsole_or_nonfloor_magnitude_integral_n_s': total,
        'maximum_nonsole_or_nonfloor_force_n': peak, 'active_raw_rows': active_rows,
        'pair_magnitude_integrals_n_s': pairs, 'physics_substeps': 4,
        'self_contact_counted_once_per_native_row': True,
        'force_threshold_applied': False, 'damage_or_net_impulse_claimed': False}
