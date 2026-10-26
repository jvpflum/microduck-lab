"""Pure shot selection and session-only USD camera authoring; no solver."""
import json,math
from pathlib import Path
CAMERA='/Root/AtticFilmCamera'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def load_plan(path):
 p=json.loads(Path(path).read_text());require(p['schema']=='attic-physical-film-shot-plan-r1'and p['fps']==25 and p['frames']==500 and p['resolution']==[1920,1080]and p['camera_path']==CAMERA,'Film format differs')
 require(len(p['shots'])==5 and p['preview_frames']==[49,149,249,349,449],'Exactly five reviewed shot midpoints required');cursor=0
 for s in p['shots']:
  require(s['start']==cursor and s['stop']==cursor+100,'Shot gaps or overlap');cursor=s['stop']
  require(20<=s['lens_mm']<=80,'Lens outside reviewed range')
  require(s['focus']in ('all','crew','duck_0'),'Unknown measured camera focus')
  for key in ('eye_offset_start_cm','eye_offset_stop_cm','target_offset_cm'):require(len(s[key])==3 and all(math.isfinite(float(x))for x in s[key]),'Nonfinite camera')
 return p

def camera_sample(plan,frame,roots_m):
 require(type(frame)is int and 0<=frame<plan['frames'],'Film frame outside plan')
 require(set(roots_m)=={f'duck_{i}'for i in range(6)}and all(len(v)==3 and all(math.isfinite(x)for x in v)for v in roots_m.values()),'Actual measured six roots required')
 shot=plan['shots'][frame//100];t=(frame-shot['start'])/(shot['stop']-shot['start']-1);t=t*t*(3-2*t)
 names=list(roots_m)if shot['focus']=='all'else [f'duck_{i}'for i in range(1,6)]if shot['focus']=='crew'else ['duck_0']
 # Exact stage units; measured root positions are native metres.
 scale=1/0.009999999776482582
 focus=[sum(roots_m[n][j]for n in names)/len(names)*scale for j in range(3)]
 eye=[focus[j]+a+(b-a)*t for j,(a,b)in enumerate(zip(shot['eye_offset_start_cm'],shot['eye_offset_stop_cm']))]
 target=[focus[j]+shot['target_offset_cm'][j]for j in range(3)];require(sum((a-b)**2 for a,b in zip(eye,target))>1,'Degenerate camera')
 return {'output_frame':frame,'shot':shot['id'],'shot_frame':frame-shot['start'],'eye_cm':eye,'target_cm':target,'focal_length_mm':shot['lens_mm'],'measured_focus_roots':names,'measured_focus_cm':focus,'actor_extrapolation':False}

def author_camera(stage,sample):
 from pxr import Gf,UsdGeom
 require(stage.GetEditTarget().GetLayer()==stage.GetSessionLayer(),'Camera must stay in anonymous session')
 camera=UsdGeom.Camera.Define(stage,CAMERA);camera.CreateProjectionAttr('perspective');camera.CreateFocalLengthAttr(sample['focal_length_mm']);camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(20.25);camera.CreateClippingRangeAttr(Gf.Vec2f(1,2000))
 matrix=Gf.Matrix4d(1).SetLookAt(Gf.Vec3d(*sample['eye_cm']),Gf.Vec3d(*sample['target_cm']),Gf.Vec3d(0,0,1)).GetInverse()
 x=UsdGeom.Xformable(camera.GetPrim());x.ClearXformOpOrder();op=x.AddTransformOp();op.Set(matrix)
 actual=x.GetLocalTransformation();require(all(abs(actual[i][j]-matrix[i][j])<1e-10 for i in range(4)for j in range(4)),'Camera readback differs')
 return {'path':CAMERA,**sample,'matrix_row_major':[float(matrix[i][j])for i in range(4)for j in range(4)],'horizontal_aperture_mm':36.,'vertical_aperture_mm':20.25,'clipping_cm':[1.,2000.]}
