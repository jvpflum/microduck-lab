"""V90 Racing candidate command composition. Keep V89 history continuous across modes."""
from pathlib import Path
import json,math
import numpy as np
from trick_controller import TrickController
class RacingActor:
 def __init__(self,path=None):
  data=json.loads(Path(path or Path(__file__).with_name('racing-weights.json')).read_text());self.weights={k:np.asarray(v,dtype=np.float32) for k,v in data.items()}
 def __call__(self,observation):
  x=np.asarray(observation,dtype=np.float32)
  if x.shape[-1]!=14 or not np.isfinite(x).all():raise ValueError('Expected finite 14-value route observation')
  w=self.weights
  for name in ('net.0','net.2'):x=np.tanh(x@w[name+'.weight'].T+w[name+'.bias'])
  return x@w['actor.weight'].T+w['actor.bias']
class Controller:
 def __init__(self,actor=None):self.actor=actor or RacingActor();self.mode='skate';self.trick=TrickController();self.resume='skate'
 def set_mode(self,mode):
  if mode not in ('skate','race'):raise ValueError('Mode must be skate or race')
  if self.trick.active:raise RuntimeError('Finish or explicitly cancel the active trick first')
  self.mode=mode
 def start_trick(self,heading,degrees,direction=1):
  self.resume=self.mode;self.trick.start(heading,direction,degrees);self.mode='trick'
 def cancel_trick(self):self.trick.cancel();self.mode=self.resume
 def update(self,heading,commands,route_observation=None,guidance=None,dt=.02):
  result=list(commands)
  if len(result)!=13 or not all(math.isfinite(v) for v in result):raise ValueError('Expected 13 finite commands')
  mode=self.resume if self.mode=='trick' else self.mode
  if mode=='race':
   if guidance is None or len(guidance)!=2 or not all(math.isfinite(v) for v in guidance):raise ValueError('Race mode requires finite forward/yaw guidance')
   u=np.tanh(self.actor(route_observation));result=[0.]*13;result[0]=float(np.clip(guidance[0]+.35*u[0],0,1));result[2]=float(np.clip(guidance[1]+.6*u[1],-1.2,1.2))
  if self.mode=='trick':
   result=self.trick.update(heading,result,dt)
   if not self.trick.active:self.mode=self.resume
  return result
