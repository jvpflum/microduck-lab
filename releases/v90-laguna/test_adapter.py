import sys,math,json,hashlib
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p))
from controller import Controller,RacingActor
from trick_controller import TrickController
actor=RacingActor();rng=np.random.default_rng(98001)
for _ in range(100):
 cmd=rng.uniform(-.5,.5,13).tolist();c=Controller(actor);assert c.update(0,cmd)==cmd
for d in (180,360,720):
 for direction in (-1,1):
  c=Controller(actor);ref=TrickController();c.start_trick(0,d,direction);ref.start(0,direction,d)
  for step in range(752):
   heading=direction*min(step*.02,3)/3*math.radians(d);cmd=[.3]+[0.]*12
   assert c.update(heading,cmd)==ref.update(heading,cmd)
  assert c.mode=='skate'
c=Controller(actor);c.set_mode('race');obs=np.zeros(14,np.float32);u=np.tanh(actor(obs));cmd=c.update(0,[0.]*13,obs,(.55,.2));assert abs(cmd[0]-np.clip(.55+.35*u[0],0,1))<1e-7
c.start_trick(0,180);assert c.mode=='trick';c.cancel_trick();assert c.mode=='race'
assert hashlib.sha256((p/'policy.onnx').read_bytes()).hexdigest()=='09663852ae1db90fdcf321dedc4e39810eb65ffc6998df3cd8fc34d3231f84f7'
print('PASS: unchanged V89 base, skating identity, exact published trick commands, race mapping, explicit handoff')
