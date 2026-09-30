"""GPU PPO course driver above frozen V89. No V89 weight updates."""
import sys,json,time,hashlib,warnings,os,copy
from pathlib import Path
import numpy as np,torch,onnxruntime as ort,mujoco,warp as wp
root=Path.home()/'microduck-lab';sys.path.insert(0,str(root/'rl-audit'))
import evaluate_mjlab_bam as ev
ev._install_project_imports(root)
sys.path.insert(0,str(Path.home()/'home-ray-envs/v91-project/tools'))
from evaluation_policy import TERM_DIMS
from cpu_skating_env import pack_history
from cuda_policy import Policy
from v89_gpu_env import Env,t
from reward_spec import GAMMA,GAE_LAMBDA,FINISH,FAILURE,TIME_COST_PER_SECOND,POTENTIAL_SCALE,self_test
BASE=Path.home()/'home-ray-envs/v91-project/reports/v89-maturity-brake-20260912/checkpoint-450.onnx'
SHA='1693fd8608037a6b73a128ba737c57adc7bcb6228cac2cc8603719f4be740c3a'

class Driver(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(14,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh());self.actor=torch.nn.Linear(64,2);self.value=torch.nn.Linear(64,1);self.logstd=torch.nn.Parameter(torch.full((2,),-.8));torch.nn.init.zeros_(self.actor.weight);torch.nn.init.zeros_(self.actor.bias)
 def forward(self,x):
  h=self.net(x);return torch.distributions.Normal(self.actor(h),self.logstd.exp()),self.value(self.critic_net(x) if hasattr(self,'critic_net') else h).squeeze(-1)

class LiveStream:
 def __init__(self,course):
  self.last=0.;self.seq=0;self.episode=0;self.phase='initializing';self.update=-1
  m=course.m;geoms=[];meshes={}
  for i in range(m.ngeom):
   body=int(m.geom_bodyid[i])
   if body==0:continue
   mesh=int(m.geom_dataid[i]);kind=int(m.geom_type[i])
   if kind==7 and mesh not in meshes:
    va=int(m.mesh_vertadr[mesh]);vn=int(m.mesh_vertnum[mesh]);fa=int(m.mesh_faceadr[mesh]);fn=int(m.mesh_facenum[mesh])
    meshes[mesh]=dict(vertices=m.mesh_vert[va:va+vn].tolist(),faces=m.mesh_face[fa:fa+fn].tolist())
   geoms.append(dict(body=body,type=kind,size=m.geom_size[i].tolist(),pos=m.geom_pos[i].tolist(),quat=m.geom_quat[i].tolist(),rgba=m.geom_rgba[i].tolist(),mesh=mesh,group=int(m.geom_group[i])))
  Path('live-model.json').write_text(json.dumps(dict(geoms=geoms,meshes=meshes,bodies=[m.body(i).name for i in range(m.nbody)])))
 def emit(self,course):
  now=time.time()
  if now-self.last<.1:return
  self.last=now;self.seq+=1
  active=(~course.done).nonzero().flatten()
  lane=int(active[0]) if len(active) else 0
  payload=dict(wall_time=now,sequence=self.seq,episode=self.episode,phase=self.phase,update=self.update,lane=lane,total_updates=course.cfg['updates'],all_done=bool(course.done.all()),sim_time=course.step_count*.02,positions=t(course.e.d.xpos)[lane].cpu().tolist(),quaternions=t(course.e.d.xquat)[lane].cpu().tolist(),gate=int(course.index[lane]),done=bool(course.done[lane]),success=bool(course.success[lane]),offtrack=bool(course.offtrack[lane]),body_contact=bool(course.body[lane]),progress_m=float(course.progress[lane]),clean_finishes=int(course.success.sum()))
  p=Path('live-frame.tmp');p.write_text(json.dumps(payload,allow_nan=False));p.replace('live-frame.json')

class Course:
 def __init__(self,n,policy,cfg):
  self.n=n;self.policy=policy;self.cfg=cfg;self.e=Env(n,'cuda:0',BASE);self.probe=self.e.probe;self.m=self.e.m;self.robot=self.e.scene['robot'].data
  self.points=torch.tensor(cfg['waypoints'],device='cuda',dtype=torch.float32);self.segments=self.points[1:]-self.points[:-1];self.lengths=self.segments.norm(dim=-1);self.cumulative=torch.cat((torch.zeros(1,device='cuda'),self.lengths.cumsum(0)))
  self.gyro=int(self.m.sensor_adr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SENSOR,'imu_ang_vel')]);self.trunk=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
  self.wheel_margin=max(cfg['wheel_edge_margin_m'],max(float(self.m.geom_rbound[i]) for i in range(self.m.ngeom) if int(self.m.geom_bodyid[i]) in self.e.wheel_bodies))
  self.cmd=torch.zeros(n,13,device='cuda');self.previous=torch.zeros(n,14,device='cuda');self.hit=torch.zeros(n,device='cuda',dtype=torch.bool);self.body=self.hit.clone();self.tilt=torch.zeros(n,device='cuda');self.driving=False
  original=type(self.probe).__call__;outer=self
  def observer(p,e):
   result=original(p,e)
   if not outer.driving:return result
   d=e.d;geom=t(d.contact.geom).long();count=t(d.nacon)[0];world=t(d.contact.worldid).long();valid=(torch.arange(d.naconmax,device='cuda')<count)&(world>=0)&(world<n)&(t(d.contact.dist)<=0)
   barrel=torch.zeros_like(valid)
   for g in e.barrel_geom_ids:barrel|=(geom==g).any(-1)
   accum=torch.zeros(n,device='cuda',dtype=torch.int32);accum.scatter_add_(0,world.clamp(0,n-1),(valid&barrel).int());outer.hit|=(accum>0)&~outer.done
   outer.body|=p.body_sensor.data.found.any(-1)&~outer.done;outer.offtrack|=outer.wheel_offtrack(e)&~outer.done;q=outer.robot.root_link_quat_w
   outer.tilt=torch.where(~outer.done,torch.maximum(outer.tilt,torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1))),outer.tilt)
   return result
  self.probe.__class__=type('CourseObserver',(type(self.probe),),{'__call__':observer})
  self.live=LiveStream(self);self.reset()
 def track_edges(self,wheels):
  d=wheels[:,:,None,:]-self.points[None,None,:-1,:]
  f=(d*self.segments).sum(-1)/self.lengths.square()
  near=self.points[:-1]+f.clamp(0,1)[...,None]*self.segments
  dist=(wheels[:,:,None,:]-near).norm(dim=-1)
  # Restrict to the current route neighborhood: a neighboring section is no shortcut.
  ids=torch.arange(len(self.segments),device='cuda')[None,:]
  delta=(ids-(self.index[:,None]-1)).abs();delta=torch.minimum(delta,len(self.segments)-delta)
  dist=dist.masked_fill(delta[:,None,:]>5,float('inf'))
  edge=dist.min(-1).values.max(-1).values+self.wheel_margin
  return edge
 def wheel_offtrack(self,e):
  edge=self.track_edges(t(e.d.xpos)[:,e.wheel_bodies,:2])
  self.max_wheel_edge=torch.where(~self.done,torch.maximum(self.max_wheel_edge,edge),self.max_wheel_edge)
  return edge>self.cfg['track_half_width_m']
 def frame(self):
  q=t(self.e.d.xquat)[:,self.trunk];g=torch.tensor([0.,0.,-1.],device='cuda').expand(self.n,-1);cross=2*torch.cross(q[:,1:],g,dim=-1);gravity=g-q[:,:1]*cross+torch.cross(q[:,1:],cross,dim=-1)
  return torch.cat((t(self.e.d.sensordata)[:,self.gyro:self.gyro+3],gravity,self.robot.joint_pos-self.e.default,self.robot.joint_vel,self.previous,self.cmd),-1)
 def packed(self):
  parts=[];offset=0
  for width in TERM_DIMS:parts.append(self.history[:,:,offset:offset+width].reshape(self.n,-1));offset+=width
  return torch.cat(parts,-1)
 def reset(self):
  self.driving=False;self.e.reset();self.cmd.zero_();self.previous.zero_();self.history=self.frame()[:,None].repeat(1,256,1)
  self.index=torch.ones(self.n,device='cuda',dtype=torch.long);self.done=torch.zeros(self.n,device='cuda',dtype=torch.bool);self.success=self.done.clone();self.offtrack=self.done.clone();self.max_wheel_edge=torch.zeros(self.n,device='cuda');self.hit.zero_();self.body.zero_();self.tilt.zero_();self.progress=torch.zeros(self.n,device='cuda');self.step_count=0;self.max_offtrack=torch.zeros(self.n,device='cuda');self.finish_time=torch.full((self.n,),-1.,device='cuda');self.driving=True;self.traces=[];self.corner_speeds=[]
  self.live.episode+=1;self.live.emit(self)
  return self.observe()[0]
 def observe(self):
  xy=self.robot.root_link_pos_w[:,:2];q=self.robot.root_link_quat_w;yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));c=yaw.cos();s=yaw.sin()
  relative=[]
  for look in range(3):
   d=self.points[(self.index+look).clamp_max(len(self.points)-1)]-xy;relative.extend(((c*d[:,0]+s*d[:,1])/3,(-s*d[:,0]+c*d[:,1])/3))
  d=xy[:,None]-self.points[:-1];projection=(d*self.segments).sum(-1)/self.lengths.square();near=self.points[:-1]+projection.clamp(0,1)[:,:,None]*self.segments;off=(xy[:,None]-near).norm(dim=-1).min(-1).values
  v=self.robot.root_link_lin_vel_w;vx=c*v[:,0]+s*v[:,1];vy=-s*v[:,0]+c*v[:,1];g=self.frame()[:,3:6]
  obs=torch.stack(relative+[vx,vy,self.robot.root_link_ang_vel_b[:,2],g[:,0],g[:,1],g[:,2],self.index.float()/len(self.points),off],-1)
  target=self.points[self.index]-xy;angle=torch.atan2(-s*target[:,0]+c*target[:,1],c*target[:,0]+s*target[:,1])
  steering=(1.2*angle).clamp(-.6,.6);throttle=(.55*(1-.65*(angle.abs()/1.57).clamp(0,1))).clamp(.15,.55)
  return obs,throttle,steering,off
 @torch.no_grad()
 def step(self,action,verify=None):
  old_done=self.done.clone();old_progress=self.progress.clone()
  for _ in range(5):
   obs,throttle,steering,off=self.observe();u=action.tanh()
   self.cmd.zero_();self.cmd[:,0]=(throttle+.35*u[:,0]).clamp(0,1.);self.cmd[:,2]=(steering+.6*u[:,1]).clamp(-1.2,1.2);self.cmd[self.done]=0
   if self.step_count<100:self.cmd.zero_()
   self.history=torch.cat((self.history[:,1:],self.frame()[:,None]),1);packed=self.packed();a=self.policy(packed)
   if verify is not None and self.step_count%50==0:
    assert np.array_equal(pack_history(self.history[:1].cpu().numpy()),packed[:1].cpu().numpy())
    expected=verify.run(None,{verify.get_inputs()[0].name:packed[:1].cpu().numpy()})[0];assert np.max(abs(expected-a[:1].cpu().numpy()))<1e-4
   self.e.step(a);self.previous=a;self.step_count+=1
   xy=self.robot.root_link_pos_w[:,:2];reached=(xy-self.points[self.index]).norm(dim=-1)<self.cfg['gate_radius_m'];reached &= ~self.done
   finish=reached&(self.index==len(self.points)-1);self.finish_time=torch.where(finish,torch.full_like(self.finish_time,self.step_count*.02),self.finish_time)
   self.index=torch.where(reached,(self.index+1).clamp_max(len(self.points)-1),self.index)
   seg=self.index-1;fraction=((xy-self.points[seg])*self.segments[seg]).sum(-1)/self.lengths[seg].square();self.progress=torch.where(~self.done,self.cumulative[seg]+fraction.clamp(0,1)*self.lengths[seg],self.progress)
   obs,_,_,off=self.observe();self.max_offtrack=torch.maximum(self.max_offtrack,off)
   failure=self.hit|self.body|self.offtrack|(self.tilt>.7854)|(off>self.cfg['track_half_width_m'])|~self.probe.finite
   self.success|=finish&~failure;self.done|=failure|finish
   if self.step_count>=int(self.cfg['episode_s']*50):self.done[:]=True
   self.live.emit(self)
   if self.step_count%10==0:self.traces.append(t(self.e.d.qpos)[0].cpu().clone())
   if bool(reached.any()):self.corner_speeds.append(dict(step=self.step_count,gates=self.index[reached].cpu().tolist(),speeds=self.robot.root_link_lin_vel_w[reached,:2].norm(dim=-1).cpu().tolist()))
  before=POTENTIAL_SCALE*(old_progress/self.cumulative[-1]).clamp(0,1)
  after=torch.where(self.done,0.,POTENTIAL_SCALE*(self.progress/self.cumulative[-1]).clamp(0,1))
  terminal=(self.done&~old_done).float()*torch.where(self.success,FINISH,FAILURE)
  reward=-TIME_COST_PER_SECOND*.1+GAMMA*after-before+terminal
  reward=torch.where(old_done,0.,reward);return obs,reward,self.done.clone(),~old_done
 def result(self):return dict(clean_finishes=int(self.success.sum()),worlds=self.n,barrel_collisions=int(self.hit.sum()),body_contacts=int(self.body.sum()),max_progress_m=float(self.progress.max()),mean_progress_m=float(self.progress.mean()),finish_times_s=torch.where(self.success,self.finish_time,-1.).cpu().tolist(),offtrack_failures=int(self.offtrack.sum()),maximum_wheel_edge_m=float(self.max_wheel_edge.max()),lap_length_m=float(self.cumulative[-1]),elevation=self.cfg['elevation'],maximum_offtrack_m=float(self.max_offtrack.max()),corner_speeds=self.corner_speeds)

@torch.no_grad()
def evaluate(course,driver,label,steps=None,verify=None):
 course.live.phase=label;obs=course.reset();steps=int(course.cfg['episode_s']*10) if steps is None else steps
 for k in range(steps):
  action=torch.zeros(course.n,2,device='cuda') if driver is None else driver(obs)[0].mean
  obs,_,_,_=course.step(action,verify)
  if bool(course.done.all()):break
 result=course.result();Path(label+'.json').write_text(json.dumps(result,indent=2));torch.save(torch.stack(course.traces),label+'-trace.pt');print(json.dumps(dict(stage=label,**{k:v for k,v in result.items() if k not in ('corner_speeds','finish_times_s')})),flush=True);return result

def main():
 cfg=json.loads(Path('course.json').read_text());smoke=os.environ.get('AGILITY_SMOKE')=='1';n=8 if smoke else cfg['worlds'];torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
 assert hashlib.sha256(BASE.read_bytes()).hexdigest()==SHA
 self_test()
 raw=Policy(BASE,'cuda').eval()
 with torch.no_grad(),warnings.catch_warnings():warnings.simplefilter('ignore');policy=torch.jit.trace(raw,torch.zeros(n,15616,device='cuda'),check_trace=False)
 reference=ort.InferenceSession(str(BASE),providers=['CPUExecutionProvider']);course=Course(n,policy,cfg)
 if smoke:
  center=torch.zeros(n,4,2,device='cuda');assert bool((course.track_edges(center)<cfg['track_half_width_m']).all())
  outside=torch.full_like(center,100.);assert bool((course.track_edges(outside)>cfg['track_half_width_m']).all())
  straddle=center.clone();straddle[:,0,1]=cfg['track_half_width_m']+.1;assert bool((course.track_edges(straddle)>cfg['track_half_width_m']).all()),'One wheel outside must fail'
  from reward_spec import scalar_reward
  for old_p,new_p,done,success in [(0.,1.,False,False),(59.,60.,True,True),(10.,11.,True,False)]:
   expected=scalar_reward(old_p,new_p,60.,False,done,success)
   actual=-.01+GAMMA*(0. if done else 20*new_p/60)-20*old_p/60+((100. if success else -100.) if done else 0.)
   assert abs(expected-actual)<1e-8
  result=evaluate(course,None,'smoke',80,reference);assert course.probe.finite.all();assert result['max_progress_m']>.25,'Baseline did not leave start';Path('summary.json').write_text(json.dumps(dict(status='smoke-passed',result=result)));return
 baseline=None;driver=Driver().cuda();driver.critic_net=copy.deepcopy(driver.net)
 assert hashlib.sha256(Path('initial-driver.pt').read_bytes()).hexdigest()=='2d1a028358803e845e58c10d22bcb80b4c9ae343f679e2b4c4d56846b2cfe3cd'
 if Path('initial-driver.pt').exists():driver.load_state_dict(torch.load('initial-driver.pt',map_location='cuda',weights_only=False)['driver'])
 anchor=copy.deepcopy(driver).eval()
 for p in anchor.parameters():p.requires_grad_(False)
 with torch.no_grad():driver.logstd.fill_(-3.)
 driver.logstd.requires_grad_(False)
 transfer=evaluate(course,anchor,'transfer-control')
 optimizer=torch.optim.Adam([p for p in driver.parameters() if p.requires_grad],lr=1e-4);obs=course.reset();logs=[]
 for update in range(cfg['updates']):
  course.live.phase='PPO training';course.live.update=update
  obs=course.reset();data=[]
  for k in range(cfg['horizon']):
   with torch.no_grad():dist,value=driver(obs);action=dist.sample();logp=dist.log_prob(action).sum(-1)
   next_obs,reward,done,valid=course.step(action);data.append((obs,action,logp,value,reward,done,valid));obs=next_obs
   if bool(done.all()):break
  with torch.no_grad():next_value=driver(obs)[1];adv=torch.zeros(n,device='cuda');advantages=[]
  for k in reversed(range(len(data))):
   _,_,_,v,r,d,valid=data[k];mask=(~d).float();adv=r+GAMMA*next_value*mask-v+GAMMA*GAE_LAMBDA*mask*adv;advantages.append(adv);next_value=v
  advantages=torch.stack(advantages[::-1]);values=torch.stack([r[3] for r in data]);returns=advantages+values
  flat=lambda i:torch.stack([r[i] for r in data]).flatten(0,1)
  valid=flat(6);x=flat(0)[valid];a=flat(1)[valid];old=flat(2)[valid];target=returns.flatten()[valid].detach();adv=advantages.flatten()[valid].detach()
  if len(x):
   adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
   kl_stop=False
   for epoch in range(4):
    if kl_stop:break
    order=torch.randperm(len(x),device='cuda')
    for ids in order.split(2048):
     dist,value=driver(x[ids]);ratio=(dist.log_prob(a[ids]).sum(-1)-old[ids]).exp();loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.9,1.1)*adv[ids]).mean()+.25*(value-target[ids]).square().mean()+.1*(dist.mean.tanh()-anchor(x[ids])[0].mean.tanh()).square().mean()
     logratio=dist.log_prob(a[ids]).sum(-1)-old[ids]
     if float(((ratio-1)-logratio).mean().detach())>.02:kl_stop=True;break
     assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(driver.parameters(),.5);optimizer.step()
  row=dict(update=update,valid_samples=len(x),mean_reward=float(flat(4).mean()),**{k:v for k,v in course.result().items() if k in ('clean_finishes','barrel_collisions','mean_progress_m','max_progress_m')});logs.append(row);Path('progress.json').write_text(json.dumps(logs,indent=2));print(json.dumps(row),flush=True)
  if (update+1)%1==0:
   torch.save(dict(driver=driver.state_dict(),optimizer=optimizer.state_dict(),config=cfg,update=update+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),f'driver-{update+1}.pt')
 trained=evaluate(course,driver,'trained')
 # Fixed held-out reset-heading condition, never used for PPO updates.
 old_q=course.e.initial.qpos[3:7].copy();angle=np.deg2rad(3.);course.e.initial.qpos[3:7]=[np.cos(angle/2),0,0,np.sin(angle/2)]
 heldout_anchor=evaluate(course,anchor,'heldout-anchor');heldout_candidate=evaluate(course,driver,'heldout-candidate');course.e.initial.qpos[3:7]=old_q
 gates=dict(clean_laps_improved=trained['clean_finishes']>=transfer['clean_finishes']+8,heldout_not_worse=heldout_candidate['clean_finishes']>=heldout_anchor['clean_finishes'],no_more_body_contacts=trained['body_contacts']<=transfer['body_contacts'] and heldout_candidate['body_contacts']<=heldout_anchor['body_contacts'],no_more_offtrack=trained['offtrack_failures']<=transfer['offtrack_failures'] and heldout_candidate['offtrack_failures']<=heldout_anchor['offtrack_failures'])
 torch.save(dict(driver=anchor.state_dict(),config=cfg,reason='Preserved starting controller'), 'protected-course-driver.pt')
 Path('qualification.json').write_text(json.dumps(dict(gates=gates,candidate_passed=all(gates.values()),heldout_anchor=heldout_anchor,heldout_candidate=heldout_candidate,automatic_promotion=False),indent=2))
 torch.save(dict(driver=driver.state_dict(),config=cfg), 'course-driver.pt')
 Path('summary.json').write_text(json.dumps(dict(status='training-complete-pending-validation',algorithm='Full-lap PPO, potential-shaped completion/time reward, separate critic, reduced anchor penalty',gpu=torch.cuda.get_device_name(),baseline=baseline,transfer_control=transfer,trained=trained,updates=cfg['updates'],worlds=n,V89_sha256=SHA,release_promoted=False,reward_version='clean-lap-v2',qualification_gates=gates,independent_validation_required=True),indent=2))
if __name__=='__main__':main()
