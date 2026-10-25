import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from isaac_workshop.compare_probe_pair import compare
from test_probe_controller import fixture


class PairTests(unittest.TestCase):
    def write(self,root,name,blocked):
        p=root/name;p.mkdir();(p/'navigation-frames').mkdir()
        depth,k,pose=fixture(blocked)
        f=p/'navigation-frames/000000.npz'
        np.savez_compressed(f,depth_m=depth,intrinsics=k,camera_to_body=pose,root_position_w_m=[-.5,-11,.15],root_quaternion_xyzw=[0,0,.707,.707],policy_observation=np.zeros(61),goal_body_m=[.9,0])
        summary={'success':True,'identity':{'navigation_case':'left' if blocked else 'clear','seed':2000,
            'owned_files':{'caller.py':'fixture'},'source_files':{'task.py':'fixture'},
            'attic':{'base':{'fixture':True},'trial_manifest_sha256':'fixture'}},
            'models':{'checkpoint_sha256':'fixture'},'model_state_unchanged':True,'autoresets_performed':0}
        row={'control_step':0,'decision':True,'raw_frame_file':'navigation-frames/000000.npz','raw_frame_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'fault':None,'command_body_twist':[0.,0.,0.] if blocked else [.15,0.,0.]}
        (p/'summary.json').write_text(json.dumps(summary));(p/'navigation.jsonl').write_text(json.dumps(row)+'\n')
        return p
    def test_pair_math_accepts_same_state_changed_depth_but_not_navigation_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.write(root,'clear',False);b=self.write(root,'blocked',True);r=compare(a,b)
            self.assertTrue(r['matched_obstacle_sensing_demonstrated']);self.assertFalse(r['learned_avoidance']);self.assertFalse(r['robust_navigation_qualified'])
    def test_changed_seed_reset_or_model_never_qualifies(self):
        for change in ('seed','reset','model','payload','base'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);a=self.write(root,'clear',False);b=self.write(root,'blocked',True)
                p=b/'summary.json';s=json.loads(p.read_text())
                if change=='seed':s['identity']['seed']=2001
                if change=='reset':s['autoresets_performed']=1
                if change=='model':s['model_state_unchanged']=False
                if change=='payload':s['identity']['owned_files']['caller.py']='changed'
                if change=='base':s['identity']['attic']['base']['fixture']=False
                p.write_text(json.dumps(s));self.assertFalse(compare(a,b)['matched_obstacle_sensing_demonstrated'])
    def test_changed_frame_bytes_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.write(root,'clear',False);b=self.write(root,'blocked',True)
            with (b/'navigation-frames/000000.npz').open('ab') as stream:stream.write(b'drift')
            with self.assertRaisesRegex(ValueError,'Raw frame changed'):compare(a,b)
    def test_camera_fault_not_conflated_with_obstacle_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.write(root,'clear',False);b=self.write(root,'blocked',True)
            p=b/'navigation.jsonl';r=json.loads(p.read_text());r['fault']='Unknown frontal depth requires stop';p.write_text(json.dumps(r)+'\n')
            self.assertFalse(compare(a,b)['matched_obstacle_sensing_demonstrated'])


if __name__=='__main__':unittest.main()
