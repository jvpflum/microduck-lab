import unittest
import numpy as np
from tools.evaluation_policy import ObservationHistory, TERM_DIMS

class HistoryContractTests(unittest.TestCase):
    def test_term_order_duplicate_read_and_reset(self):
        h=ObservationHistory(256)
        a=np.arange(61,dtype=np.float32);b=a+100
        first=h.observe(a,0)
        self.assertEqual(first.shape,(15616,))
        np.testing.assert_array_equal(first[:768].reshape(256,3),np.repeat(a[None,:3],256,axis=0))
        second=h.observe(b,.02)
        np.testing.assert_array_equal(second[:768].reshape(256,3)[-1],b[:3])
        np.testing.assert_array_equal(h.observe(b+10,.02),second)
        with self.assertRaises(ValueError):h.observe(a,0)
        h.reset();np.testing.assert_array_equal(h.observe(a,0),first)
        offset=0;frame_offset=0
        for width in TERM_DIMS:
            term=second[offset:offset+256*width].reshape(256,width)
            np.testing.assert_array_equal(term[-1],b[frame_offset:frame_offset+width])
            offset+=256*width;frame_offset+=width
