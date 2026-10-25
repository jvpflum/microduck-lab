"""RNG state preservation and safe owned-reload source guards; no training."""
import ast, copy, json, unittest
from pathlib import Path
import numpy as np
from checkpoint_state import encode_numpy_rng, decode_numpy_rng

class SafeCheckpointTests(unittest.TestCase):
    def test_roundtrip_preserves_future_uniform_and_cached_normal(self):
        a=np.random.RandomState(19231);a.normal(size=3)
        encoded=json.loads(json.dumps(encode_numpy_rng(a.get_state())))
        b=np.random.RandomState();b.set_state(decode_numpy_rng(encoded))
        np.testing.assert_array_equal(a.normal(size=97),b.normal(size=97))
        np.testing.assert_array_equal(a.uniform(size=97),b.uniform(size=97))
    def test_malformed_state_rejected(self):
        source=encode_numpy_rng(np.random.RandomState(7).get_state())
        for key,value in [('generator','other'),('keys',[0]*623),('position',625),('has_gauss',2),('cached',float('nan'))]:
            bad=copy.deepcopy(source);bad[key]=value
            with self.assertRaises(ValueError):decode_numpy_rng(bad)
        bad=copy.deepcopy(source);bad['keys'][0]=-1
        with self.assertRaises(ValueError):decode_numpy_rng(bad)
    def test_all_torch_loads_tensor_safe_and_no_external_checkpoint_cli(self):
        tree=ast.parse(Path(__file__).with_name('train.py').read_text())
        loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load']
        self.assertEqual(len(loads),1)
        self.assertTrue(next(k.value.value for k in loads[0].keywords if k.arg=='weights_only'))
        self.assertFalse(any(isinstance(n, ast.Constant) and n.value == '--checkpoint' for n in ast.walk(tree)))

if __name__=='__main__':unittest.main()
