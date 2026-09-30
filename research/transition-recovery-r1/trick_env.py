"""Backward turn command-policy PPO; V89 and physics remain frozen."""
import sys,json,hashlib,warnings,os
from pathlib import Path
import numpy as np,torch,mujoco,onnxruntime as ort
root=Path.home()/'microduck-lab';sys.path.insert(0,str(root/'rl-audit'))
import evaluate_mjlab_bam as ev
ev._install_project_imports(root)
sys.path.insert(0,str(Path.home()/'home-ray-envs/v91-project/tools'))
from evaluation_policy import TERM_DIMS
from cpu_skating_env import pack_history
from cuda_policy import Policy
from v89_gpu_env import Env,t
BASE=Path.home()/'home-ray-envs/v91-project/reports/v89-maturity-brake-20260912/checkpoint-450.onnx'
SHA='1693fd8608037a6b73a128ba737c57adc7bcb6228cac2cc8603719f4be740c3a'
class Driver(torch.nn.Module):
 def __init__(self):
  super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(14,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,2));self.critic=torch.nn.Sequential(torch.nn.Linear(14,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1));torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)
 def forward(self,x):return torch.distributions.Normal(self.actor(x),torch.full((2,),.12,device='cuda')),self.critic(x).squeeze(-1)
class World:
 def __init__(self,n,policy):
  self.n=n;self.policy=policy;self.e=Env(n,'cuda:0',BASE);self.robot=self.e.scene['robot'].data
  self.gyro=int(self.e.m.sensor_adr[mujoco.mj_name2id(self.e.m,mujoco.mjtObj.mjOBJ_SENSOR,'imu_ang_vel')]);self.trunk=mujoco.mj_name2id(self.e.m,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
  self.target=torch.tensor([np.deg2rad(d)*s for d in (180,360,720) for s in (-1,1)],device='cuda',dtype=torch.float32).repeat(n//6);assert len(self.target)==n
  self.cmd=torch.zeros(n,13,device='cuda');self.previous=torch.zeros(n,14,device='cuda');self.driving=False
  original=type(self.e.probe).__call__;outer=self
  def observer(p,e):
   z=original(p,e)
   if outer.driving:
    outer.bad|=p.body_sensor.data.found.any(-1)|~p.finite
    q=outer.robot.root_link_quat_w;outer.tilt=torch.maximum(outer.tilt,torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    outer.bad|=outer.tilt>np.pi/4
   return z
  self.e.probe.__class__=type('BackwardObserver',(type(self.e.probe),),{'__call__':observer});self.reset()
 def frame(self):
  q=t(self.e.d.xquat)[:,self.trunk];g=torch.tensor([0.,0.,-1.],device='cuda').expand(self.n,-1);cross=2*torch.cross(q[:,1:],g,dim=-1);gravity=g-q[:,:1]*cross+torch.cross(q[:,1:],cross,dim=-1)
  return torch.cat((t(self.e.d.sensordata)[:,self.gyro:self.gyro+3],gravity,self.robot.joint_pos-self.e.default,self.robot.joint_vel,self.previous,self.cmd),-1)
 def yaw(self):
  q=self.robot.root_link_quat_w;return torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
 def velocity(self):
  yaw=self.yaw();v=self.robot.root_link_lin_vel_w;return v[:,0]*yaw.cos()+v[:,1]*yaw.sin(),-v[:,0]*yaw.sin()+v[:,1]*yaw.cos()
 def reset(self):
  self.driving=False;self.e.reset();self.cmd.zero_();self.previous.zero_();self.k=0;self.rotation=torch.zeros(self.n,device='cuda');self.lastyaw=self.yaw().clone();self.bad=torch.zeros(self.n,device='cuda',dtype=torch.bool);self.tilt=torch.zeros(self.n,device='cuda');self.history=self.frame()[:,None].repeat(1,256,1);self.entry=[];self.exit=[];self.reverse=[];self.trace=[];self.driving=True
  return self.observe()
 def observe(self):
  vx,vy=self.velocity();error=self.target-self.rotation;g=self.frame()[:,3:6]
  return torch.stack((vx,vy,self.robot.root_link_ang_vel_b[:,2],g[:,0],g[:,1],g[:,2],error/np.pi,self.rotation/np.pi,self.target/np.pi,torch.full_like(vx,self.k/1250),self.cmd[:,0],self.cmd[:,2],error.sin(),error.cos()),-1)
 def result(self):
  entry=torch.stack(self.entry).mean(0) if self.entry else torch.zeros(self.n,device='cuda');tail=torch.stack(self.exit) if self.exit else torch.zeros(1,self.n,3,device='cuda');fraction=torch.stack(self.reverse).float().mean(0) if self.reverse else torch.zeros(self.n,device='cuda')
  gates=dict(backward_entry=entry<-.1,rotation_complete=(self.target-self.rotation).abs()<np.deg2rad(20),no_body_contact_or_fall=~self.bad,backward_through_turn=fraction>=.8,backward_exit=tail[:,:,0].mean(0)<-.1,settled_yaw=tail[:,:,1].abs().mean(0)<.5,stable_heading=tail[:,:,2].abs().max(0).values<np.deg2rad(20))
  passed=torch.stack(list(gates.values())).all(0);rows=[]
  for i in range(self.n):rows.append(dict(degrees=int(round(abs(float(self.target[i]))*180/np.pi)),direction=1 if self.target[i]>0 else -1,passed=bool(passed[i]),gates={k:bool(v[i]) for k,v in gates.items()},entry_vx=float(entry[i]),exit_vx=float(tail[:,i,0].mean()),backward_fraction=float(fraction[i]),rotation_deg=float(self.rotation[i]*180/np.pi)))
  return dict(passed=int(passed.sum()),worlds=self.n,cases=rows)
 @torch.no_grad()
 def step(self,residual,straight=False,reference=None):
  oldbad=self.bad.clone();olderror=(self.target-self.rotation).abs()
  for _ in range(5):
   self.cmd.zero_()
   if self.k>=100:
    self.cmd[:,0]=-.3
    if self.k>=250 and not straight:
     u=residual.tanh();error=self.target-self.rotation;self.cmd[:,0]=(-.3+.15*u[:,0]).clamp(-.8,-.05);self.cmd[:,2]=(.7*error+.3*u[:,1]).clamp(-.6,.6)
   self.cmd[self.bad]=0
   self.history=torch.cat((self.history[:,1:],self.frame()[:,None]),1);parts=[];offset=0
   for width in TERM_DIMS:parts.append(self.history[:,:,offset:offset+width].reshape(self.n,-1));offset+=width
   packed=torch.cat(parts,-1);a=self.policy(packed)
   if reference is not None and self.k%50==0:
    assert np.array_equal(pack_history(self.history[:1].cpu().numpy()),packed[:1].cpu().numpy());expected=reference.run(None,{reference.get_inputs()[0].name:packed[:1].cpu().numpy()})[0];assert np.max(abs(expected-a[:1].cpu().numpy()))<1e-4
   self.e.step(a);self.previous=a;now=self.yaw();delta=now-self.lastyaw
   if self.k>=250:self.rotation+=torch.atan2(delta.sin(),delta.cos())
   self.lastyaw=now.clone();self.k+=1;vx,vy=self.velocity()
   if 200<=self.k<250:self.entry.append(vx.clone())
   if self.k>=250:self.reverse.append(vx<-.05)
   if self.k>=1150:self.exit.append(torch.stack((vx,self.robot.root_link_ang_vel_b[:,2],self.target-self.rotation),-1))
   if self.k%10==0:self.trace.append(t(self.e.d.qpos).cpu().clone())
  vx,vy=self.velocity();reward=2*(olderror-(self.target-self.rotation).abs())+.1*(-vx).clamp(-.5,.5)-.05*vy.abs()-.01
  reward-=10*(self.bad&~oldbad).float()
  done=self.bad.clone()
  if self.k>=1250:
   passed=torch.tensor([r['passed'] for r in self.result()['cases']],device='cuda');reward+=torch.where(passed,30.,-10.);done[:]=True
  return self.observe(),torch.where(oldbad,0.,reward),done,~oldbad
@torch.no_grad()
def evaluate(w,driver,label):
 obs=w.reset()
 for _ in range(250):obs,_,_,_=w.step(driver(obs)[0].mean if driver else torch.zeros(w.n,2,device='cuda'))
 result=w.result();Path(label+'.json').write_text(json.dumps(result,indent=2));torch.save(torch.stack(w.trace),label+'-trace.pt');return result

def main():
 cfg=json.loads(Path('course.json').read_text());smoke=os.getenv('AGILITY_SMOKE')=='1';n=12 if smoke else 48;torch.manual_seed(cfg['seed']);torch.set_num_threads(2);torch.cuda.set_device(0)
 assert hashlib.sha256(BASE.read_bytes()).hexdigest()==SHA
 raw=Policy(BASE,'cuda').eval()
 with torch.no_grad(),warnings.catch_warnings():warnings.simplefilter('ignore');policy=torch.jit.trace(raw,torch.zeros(n,15616,device='cuda'),check_trace=False)
 w=World(n,policy);reference=ort.InferenceSession(str(BASE),providers=['CPUExecutionProvider'])
 for _ in range(80):w.step(torch.zeros(n,2,device='cuda'),straight=True,reference=reference if smoke else None)
 vx,_=w.velocity();entry=torch.stack(w.entry).mean(0);ready=(entry<-.1)&~w.bad
 pre=dict(worlds=n,backward_ready=int(ready.sum()),mean_entry_vx=float(entry.mean()),failed=int(w.bad.sum()),finite=bool(w.e.probe.finite.all()));Path('backward-preflight.json').write_text(json.dumps(pre,indent=2));assert w.e.probe.finite.all()
 if smoke:Path('summary.json').write_text(json.dumps(dict(status='execution-smoke-passed',backward_preflight=pre)));return
 if int(ready.sum())<int(.8*n):Path('summary.json').write_text(json.dumps(dict(status='blocked-backward-prerequisite',backward_preflight=pre,training_started=False,release_promoted=False)));return
 baseline=evaluate(w,None,'baseline');driver=Driver().cuda();optimizer=torch.optim.Adam(driver.parameters(),lr=1e-4);logs=[]
 for update in range(cfg['updates']):
  obs=w.reset();data=[]
  for _ in range(250):
   with torch.no_grad():dist,value=driver(obs);action=dist.sample();logp=dist.log_prob(action).sum(-1)
   nxt,reward,done,valid=w.step(action);data.append((obs,action,logp,value,reward,done,valid));obs=nxt
  adv=torch.zeros(n,device='cuda');nxtv=adv.clone();advantages=[]
  for row in reversed(data):
   _,_,_,v,r,d,_=row;mask=(~d).float();adv=r+.999*nxtv*mask-v+.999*.98*mask*adv;advantages.append(adv);nxtv=v
  aa=torch.stack(advantages[::-1]);returns=aa+torch.stack([r[3] for r in data]);flat=lambda i:torch.stack([r[i] for r in data]).flatten(0,1);valid=flat(6);x=flat(0)[valid];actions=flat(1)[valid];old=flat(2)[valid];target=returns.flatten()[valid];advantages=aa.flatten()[valid]
  if len(x):
   advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
   for epoch in range(4):
    for ids in torch.randperm(len(x),device='cuda').split(2048):
     dist,value=driver(x[ids]);ratio=(dist.log_prob(actions[ids]).sum(-1)-old[ids]).exp();loss=-torch.minimum(ratio*advantages[ids],ratio.clamp(.9,1.1)*advantages[ids]).mean()+.25*(value-target[ids]).square().mean()+.02*dist.mean.square().mean();assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(driver.parameters(),.5);optimizer.step()
  logs.append(dict(update=update,passed=w.result()['passed'],worlds=n));Path('progress.json').write_text(json.dumps(logs));print(json.dumps(logs[-1]),flush=True)
  if (update+1)%10==0:torch.save(dict(driver=driver.state_dict(),optimizer=optimizer.state_dict(),update=update+1),f'driver-{update+1}.pt')
 trained=evaluate(w,driver,'trained');torch.save(dict(driver=driver.state_dict()),'backward-driver.pt');Path('summary.json').write_text(json.dumps(dict(status='training-complete-pending-independent-validation',baseline=baseline,trained=trained,release_promoted=False,V89_sha256=SHA),indent=2))
if __name__=='__main__':main()
