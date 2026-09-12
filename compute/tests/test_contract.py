import tempfile
import unittest
from pathlib import Path
from home_ray.spec import validate
from home_ray.server import Broker

class FakeTask:
    def __call__(self, f): return self
class FakeRay:
    def remote(self, **kw): return FakeTask()

class ContractTests(unittest.TestCase):
    def test_full_gpu_and_retry_guard(self):
        for patch in ({'gpus':.5},{'system_retries':1},{'env':{'CUDA_VISIBLE_DEVICES':'1'}},{'files':{'../escape':'x'}},{'cpus':float('nan')}):
            with self.assertRaises(ValueError): validate(dict(command=['true'],**patch))
        self.assertEqual(validate({'command':['true'],'gpus':1,'system_retries':2,'idempotent':True})['system_retries'],2)

    def test_batch_atomic_and_restart_marks_unknown_work(self):
        with tempfile.TemporaryDirectory() as d:
            b=Broker(d,FakeRay(),'http://unused','token')
            with self.assertRaises(ValueError): b.submit([{'command':['true']},{'command':[]}])
            self.assertEqual(b.jobs(),[])
            j=b.submit([{'command':['true']}])[0]
            j.update(state='RUNNING',attempt=1)
            b.save(j)
            b.db.close()
            b=Broker(d,FakeRay(),'http://unused','token')
            self.assertEqual(b.get(j['id'])['state'],'INTERRUPTED')
            b.db.close()

    def test_log_upload_is_idempotent_and_no_gaps(self):
        import base64
        with tempfile.TemporaryDirectory() as d:
            b=Broker(d,FakeRay(),'http://unused','token')
            j=b.submit([{'command':['true']}])[0]
            j.update(state='RUNNING',attempt=1,attempts=[{}])
            b.save(j)
            data={'stream':'stdout','offset':0,'data':base64.b64encode(b'abc').decode()}
            b.internal(j['id'],1,'log',data)
            b.internal(j['id'],1,'log',data)
            self.assertEqual((Path(d)/j['id']/'1/stdout.log').read_bytes(),b'abc')
            with self.assertRaises(ValueError): b.internal(j['id'],1,'log',dict(data,offset=10))
            b.db.close()

if __name__=='__main__': unittest.main()
