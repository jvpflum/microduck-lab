"""Predeclared paired speed advancement with completion and safety retention."""
import statistics


def compare(anchor, candidate, stress_only=False):
    assert anchor['worlds'] == candidate['worlds']
    paired = [(a, b) for a, b in zip(anchor['finish_times_s'], candidate['finish_times_s']) if a > 0 and b > 0]
    assert len(anchor['finish_times_s']) == len(candidate['finish_times_s']) == anchor['worlds']
    ratios = [b / a for a, b in paired]
    median = statistics.median(ratios) if ratios else None
    wins = sum(b < a for a, b in paired) / len(paired) if paired else 0
    gates = {'clean_laps_retained': candidate['clean_finishes'] >= anchor['clean_finishes']}
    for field in ('offtrack_failures', 'body_contacts', 'falls', 'finite_failures', 'barrel_collisions'):
        gates[field + '_retained'] = candidate[field] <= anchor[field]
    if not stress_only:
        gates.update(enough_paired_laps=len(paired) >= 16,
                     paired_median_5pct_faster=median is not None and median <= .95,
                     paired_wins_at_least_60pct=wins >= .6)
    return dict(gates=gates, passed=all(gates.values()), paired_clean_laps=len(paired),
                paired_median_time_ratio=median, paired_win_fraction=wins,
                stress_only=stress_only)
