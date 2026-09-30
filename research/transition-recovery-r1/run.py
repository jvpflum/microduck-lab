"""Fixed matched GPU diagnostic; frozen skating/tricks/racing; no optimization."""
from race_env import *
from trick_env import World
from controller import RacingActor
from trick_controller import TrickController
from v89_gpu_env import mw,wp

@torch.no_grad()
def trial(w,policy,driver,arm,seed):
 n=w.n;torch.manual_seed(seed);w.reset()
 # Explicit, matched reset-only joint perturbations. No later state corrections.
 q=t(w.e.d.qpos);q[:,w.e.qpi]+=torch.empty((n,14),device='cuda').uniform_(-.005,.005)
 mw.forward(w.e.gm,w.e.d);wp.synchronize()
 w.history=w.frame()[:,None].repeat(1,256,1)
 initial=q.cpu().numpy().tobytes();reset_hash=hashlib.sha256(initial).hexdigest()
 controls=[TrickController() for _ in range(n)]
 degrees=[180,180,360,360,720,720]*(n//6);directions=[-1,1]*(n//2)
 target=torch.tensor([d*s*np.pi/180 for d,s in zip(degrees,directions)],device='cuda')
 angle=torch.zeros(n,device='cuda');previous=w.yaw().clone()
 windows={'original_trick_tail':[],'recovery_tail':[],'race_tail':[],'skate_exit':[]}
 bad_phase={};trace=[];ready=torch.zeros(n,device='cuda',dtype=torch.bool);race_cmd=torch.zeros(n,13,device='cuda')
 for k in range(1800):
  yaw=w.yaw();vx,vy=w.velocity();cmd=torch.zeros(n,13,device='cuda');cmd[:,0]=.3 if k>=100 else 0.
  if 250<=k<1002:
   rows=[]
   for i,c in enumerate(controls):
    if k==250:c.start(float(yaw[i]),directions[i],degrees[i])
    rows.append(c.update(float(yaw[i]),cmd[i].cpu().tolist()))
   cmd=torch.tensor(rows,device='cuda')
  # Recovery is an explicit eight-second extension, not a lowered original gate.
  recovery=torch.zeros_like(cmd);recovery[:,0]=.45;recovery[:,2]=(.7*(target-angle)).clamp(-.6,.6)
  if arm=='recovery' and 1002<=k<1400:cmd=recovery
  if k==1400:
   a=torch.stack(windows['recovery_tail']);ready=(a[:,:,0].mean(0)>.15)&(a[:,:,1].abs().mean(0)<.5)&(a[:,:,2].abs().max(0).values<np.deg2rad(20))&~w.bad
  race_active=(arm=='current' and 1002<=k<1650) or (arm=='recovery' and 1400<=k<1650)
  if race_active:
   if k%5==0 or k==1002:
    obs=torch.zeros(n,14,device='cuda');obs[:,0]=1/3;obs[:,2]=.5;obs[:,4]=2/3;obs[:,6]=vx;obs[:,7]=vy;obs[:,8]=w.robot.root_link_ang_vel_b[:,2];obs[:,9:12]=w.frame()[:,3:6]
    u=driver(obs)[0].mean.tanh();race_cmd.zero_();race_cmd[:,0]=(.3+.35*u[:,0]).clamp(0,1);race_cmd[:,2]=(.6*u[:,1]).clamp(-1.2,1.2)
   if arm=='recovery':
    blend=min(1.,(k-1400)/100);cmd=torch.where(ready[:,None],(1-blend)*recovery+blend*race_cmd,recovery)
   else:cmd=race_cmd.clone()
  w.cmd.copy_(cmd);w.history=torch.cat((w.history[:,1:],w.frame()[:,None]),1)
  parts=[];offset=0
  for width in TERM_DIMS:parts.append(w.history[:,:,offset:offset+width].reshape(n,-1));offset+=width
  action=policy(torch.cat(parts,-1));w.e.step(action);w.previous=action
  now=w.yaw();delta=now-previous
  if k>=250:angle+=torch.atan2(delta.sin(),delta.cos())
  previous=now.clone();vx,vy=w.velocity();sample=torch.stack((vx,w.robot.root_link_ang_vel_b[:,2],target-angle),-1)
  for name,start,end in [('original_trick_tail',900,1000),('recovery_tail',1300,1400),('race_tail',1550,1650),('skate_exit',1700,1800)]:
   if start<=k<end:windows[name].append(sample.clone())
  if k in (249,1001,1399,1649,1799):bad_phase[str(k+1)]=w.bad.cpu().tolist()
  if k%10==0:trace.append(t(w.e.d.qpos).cpu().clone())
  if k%250==0:Path('progress.json').write_text(json.dumps(dict(arm=arm,seed=seed,step=k,total_steps=1800,gpu=torch.cuda.get_device_name())))
 stats={}
 for name,rows in windows.items():
  a=torch.stack(rows);stats[name]=dict(mean_vx=a[:,:,0].mean(0).cpu().tolist(),mean_abs_yaw=a[:,:,1].abs().mean(0).cpu().tolist(),max_heading_error_deg=(a[:,:,2].abs().max(0).values*180/np.pi).cpu().tolist())
 rows=[]
 for i in range(n):
  checks={}
  for name in ('original_trick_tail','recovery_tail'):
   s=stats[name];checks[name]=s['mean_vx'][i]>.15 and s['mean_abs_yaw'][i]<.5 and s['max_heading_error_deg'][i]<20
  checks['no_body_contact_or_fall']=not bool(w.bad[i]);checks['skate_exit']=stats['skate_exit']['mean_vx'][i]>.1
  checks['race_engaged']=bool(ready[i]) if arm=='recovery' else arm=='current'
  passed=checks['no_body_contact_or_fall'] and checks['skate_exit'] and checks['race_engaged'] and checks['recovery_tail' if arm=='recovery' else 'original_trick_tail']
  rows.append(dict(degrees=degrees[i],direction=directions[i],passed=passed,gates=checks))
 result=dict(arm=arm,reset_seed=seed,reset_sha256=reset_hash,worlds=n,passed=sum(r['passed'] for r in rows),cases=rows,phase_metrics=stats,bad_by_phase=bad_phase,scope='36-second open-floor diagnostic; synthetic straight route; not track rejoining',recovery_allowance_seconds=8 if arm=='recovery' else 0)
 name=f'{seed}-{arm}';Path(name+'.json').write_text(json.dumps(result,indent=2));torch.save(torch.stack(trace),name+'-trace.pt');print(json.dumps({k:result[k] for k in ('arm','reset_seed','worlds','passed')}),flush=True);return result

@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(98011)
 assert hashlib.sha256(BASE.read_bytes()).hexdigest()==SHA
 raw=Policy(BASE,'cuda').eval();n=12 if os.getenv('AGILITY_SMOKE')=='1' else 48
 with warnings.catch_warnings():warnings.simplefilter('ignore');policy=torch.jit.trace(raw,torch.zeros(n,15616,device='cuda'),check_trace=False)
 driver=Driver().cuda();driver.critic_net=copy.deepcopy(driver.net);driver.load_state_dict(torch.load('racing-source.pt',map_location='cuda',weights_only=False)['driver']);driver.eval()
 x=torch.randn(100,14,device='cuda');err=float((torch.tensor(RacingActor()(x.cpu().numpy()),device='cuda')-driver(x)[0].mean).abs().max());assert err<1e-5
 if os.getenv('AGILITY_SMOKE')=='1':
  cfg=json.loads(Path('course.json').read_text());course=Course(n,policy,cfg);ref=ort.InferenceSession(str(BASE),providers=['CPUExecutionProvider']);evaluate(course,None,'smoke',80,ref);assert course.probe.finite.all();Path('summary.json').write_text(json.dumps(dict(status='smoke-passed',actor_error=err)));return
 w=World(n,policy);results=[]
 for seed in (98011,98012):
  cohort=[trial(w,policy,driver,arm,seed) for arm in ('skate_only','current','recovery')];assert len({r['reset_sha256'] for r in cohort})==1;results.extend(cohort)
 passed=all(r['passed']==n for r in results if r['arm']=='recovery')
 Path('summary.json').write_text(json.dumps(dict(status='diagnostic-complete',recovery_all_passed=passed,results=results,release_promoted=False,next_gate='Fresh course and standalone skating/trick validation required even if recovery passes',actor_error=err),indent=2))
if __name__=='__main__':main()
