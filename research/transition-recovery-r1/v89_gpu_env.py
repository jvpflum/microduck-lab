"""V89's original MuJoCo model on CUDA; no BAM actuator substitution."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,numpy as np,torch,mujoco,warp as wp,mujoco_warp as mw
sys.path.insert(0,str(Path.home()/'home-ray-envs/v91-project/tools'))
from benchmark_cpu_control_batch import ExactCpuBatch

def t(x):return wp.to_torch(x)

class RobotData:
    def __init__(self,e):self.e=e
    @property
    def default_joint_pos(self):return self.e.default[None].expand(self.e.n,-1)
    @property
    def joint_pos(self):return t(self.e.d.qpos)[:,self.e.qpi]
    @property
    def joint_vel(self):return t(self.e.d.qvel)[:,self.e.qvi]
    @property
    def root_link_pos_w(self):return t(self.e.d.qpos)[:,:3]
    @property
    def root_link_quat_w(self):return t(self.e.d.qpos)[:,3:7]
    @property
    def root_link_lin_vel_w(self):return t(self.e.d.qvel)[:,:3]
    @property
    def root_link_ang_vel_b(self):return t(self.e.d.qvel)[:,3:6]
    @property
    def site_pos_w(self):return t(self.e.d.site_xpos)

class Probe:
    def __init__(self,e):
        self.e=e;n=e.n
        self.head_site_ids=[mujoco.mj_name2id(e.m,mujoco.mjtObj.mjOBJ_SITE,name) for name in ('mouth_tip','tof','head_camera')]
        assert min(self.head_site_ids)>=0
        self.body_ids=[i for i in range(1,e.m.nbody) if i not in e.wheel_bodies]
        self.body_sensor=NS(primary_names=[e.m.body(i).name for i in self.body_ids],data=NS(found=torch.zeros(n,len(self.body_ids),device=e.device)))
        self.tire_sensor=NS(data=NS(found=torch.zeros(n,4,device=e.device)))
        self.feet_sensor=NS(data=NS(found=torch.zeros(n,2,device=e.device)))
        self.reset()
    def reset(self):self.finite=torch.ones(self.e.n,device=self.e.device,dtype=torch.bool)
    def __call__(self,e):
        d=e.d;n=e.n
        self.finite &= torch.isfinite(t(d.qpos)).all(-1)&torch.isfinite(t(d.qvel)).all(-1)
        count=t(d.nacon)[0];assert int(count)<d.naconmax,'GPU contact buffer exhausted'
        geom=t(d.contact.geom).long();world=t(d.contact.worldid).long()
        valid=(torch.arange(d.naconmax,device=e.device)<count)&(world>=0)&(world<n)&(geom>=0).all(-1)&(t(d.contact.dist)<=0)
        bodies=e.geom_bodies[geom.clamp_min(0)]
        valid &= (bodies[:,0]==0)^(bodies[:,1]==0)
        body=torch.maximum(bodies[:,0],bodies[:,1])
        flat=torch.zeros(n*e.m.nbody,device=e.device,dtype=torch.int32)
        idx=world.clamp(0,n-1)*e.m.nbody+body
        flat.scatter_add_(0,idx,valid.int());found=flat.reshape(n,e.m.nbody)>0
        self.body_sensor.data.found=found[:,self.body_ids]
        self.tire_sensor.data.found=found[:,e.wheel_bodies]
        self.feet_sensor.data.found=torch.stack([found[:,ids].any(-1) for ids in e.side_bodies],-1)
        return torch.zeros(n,device=e.device)

class Env:
    def __init__(self,n,device,base):
        wp.init();wp.set_device(device);self.n=n;self.device=device
        b=ExactCpuBatch(1,1,base);self.m=b.model;self.initial=b.data[0];self.default=torch.tensor(b.default,device=device);self.qpi=list(b.qpos_indices);self.qvi=list(b.qvel_indices);self.torque=b.torque_limit;b.close()
        # Compile physical barrels into the same V89 model before uploading it.
        import json
        from evaluate_swizzle import ROLLER_SCENE
        cfg=json.loads(Path('course.json').read_text())
        spec=mujoco.MjSpec.from_file(str(ROLLER_SCENE))
        for i,barrel in enumerate(cfg['barrels']):
            spec.worldbody.add_geom(name=f'course_barrel_{i}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[barrel['x'],barrel['y'],barrel['height']/2],size=[barrel['radius'],barrel['height']/2,0],rgba=[.95,.45,.08,1],contype=1,conaffinity=1)
        source=self.m;initial=self.initial;self.m=spec.compile()
        assert (source.nq,source.nv,source.nu)==(self.m.nq,self.m.nv,self.m.nu)
        self.m.actuator_forcerange[:]=source.actuator_forcerange
        self.m.actuator_forcelimited[:]=source.actuator_forcelimited
        self.m.dof_frictionloss[:]=source.dof_frictionloss
        self.initial=mujoco.MjData(self.m);self.initial.qpos[:]=initial.qpos;self.initial.ctrl[:]=initial.ctrl
        self.m.opt.timestep=.005
        mujoco.mj_forward(self.m,self.initial)
        self.barrel_geom_ids=[i for i in range(self.m.ngeom) if self.m.geom(i).name.startswith('course_barrel_')]

        self.qpi=torch.tensor(self.qpi,device=device);self.qvi=torch.tensor(self.qvi,device=device)
        self.wheel_bodies=sorted({int(self.m.jnt_bodyid[j]) for j in range(self.m.njnt) if self.m.joint(j).name.startswith('passive_') and 'wheel' in self.m.joint(j).name})
        assert len(self.wheel_bodies)==4
        self.side_bodies=[[int(self.m.jnt_bodyid[j]) for j in range(self.m.njnt) if self.m.joint(j).name.startswith(prefix) and 'wheel' in self.m.joint(j).name] for prefix in ('passive_L','passive_R')]
        assert all(len(x)==2 for x in self.side_bodies),self.side_bodies
        self.geom_bodies=torch.tensor(self.m.geom_bodyid,device=device,dtype=torch.long)
        self.gm=mw.put_model(self.m)
        self.d=None;self.scene={'robot':NS(data=RobotData(self))}
        class Scene(dict):pass
        self.scene=Scene(self.scene);self.scene.terrain=NS(env_origins=torch.zeros(n,3,device=device))
        self.probe=Probe(self);self.reset()
    def reset(self,seed=None):
        if seed is not None:torch.manual_seed(seed)
        self.d=mw.put_data(self.m,self.initial,nworld=self.n,nconmax=256,njmax=2048)
        self.sim=NS(mj_model=self.m,data=self.d)
        mw.forward(self.gm,self.d)
        wp.synchronize()
        self.probe.reset();self.probe(self)
        # Capture only physics, leaving controller and contact checks on CUDA Torch.
        with wp.ScopedCapture() as cap:mw.step(self.gm,self.d)
        self.graph=cap.graph
    def step(self,action):
        t(self.d.ctrl).copy_(self.default+action)
        torch.cuda.synchronize()
        for _ in range(4):
            wp.capture_launch(self.graph)
            wp.synchronize()
            self.probe(self)
            torch.cuda.synchronize()
    def close(self):pass
    def contract(self):
        assert float(self.m.opt.timestep)==.005
        assert np.allclose(self.m.actuator_forcerange,[-self.torque,self.torque])
        assert float(t(self.d.qvel).abs().max())<1e-7
        return dict(backend='original V89 MuJoCo model on mujoco-warp CUDA',physics_dt_s=.005,control_dt_s=.02,current_limit_a=1.75,hard_torque_ceiling_nm=self.torque,wheel_frictionloss=.003,entry_speed_mps=0.,model_nq=self.m.nq,model_nv=self.m.nv)
