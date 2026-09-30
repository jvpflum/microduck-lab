import unittest
from qualification import compare


def result(times, **extra):
    return dict(worlds=len(times), finish_times_s=times,
                clean_finishes=sum(t > 0 for t in times), offtrack_failures=0,
                body_contacts=0, falls=0, finite_failures=0, barrel_collisions=0, **extra)


class QualificationTests(unittest.TestCase):
    def test_faster_clean_retained_laps_pass(self):
        self.assertTrue(compare(result([100] * 20), result([94] * 20))['passed'])

    def test_fast_survivors_do_not_hide_failures(self):
        self.assertFalse(compare(result([100] * 20), result([80] * 19 + [-1]))['passed'])

    def test_speed_does_not_hide_a_fall(self):
        bad = result([90] * 20); bad['falls'] = 1
        self.assertFalse(compare(result([100] * 20), bad)['passed'])

    def test_too_few_paired_cases_cannot_pass(self):
        self.assertFalse(compare(result([100] * 8), result([80] * 8))['passed'])

    def test_unchanged_control_is_retention_not_advancement(self):
        a = result([100] * 20)
        self.assertFalse(compare(a, a)['passed'])
        self.assertTrue(compare(a, a, stress_only=True)['passed'])


if __name__ == '__main__':
    unittest.main()
