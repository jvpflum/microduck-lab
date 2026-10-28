"""Synthetic fixtures, not native render/physics evidence."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from isaac_workshop import seasonal_theme as t

ROOT=Path(__file__).resolve().parents[1]
class ThemeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.config,self.source,self.assets=[self.root/p for p in ('config','source','assets')]
  for p in (self.config/'profiles',self.source,self.assets):p.mkdir(parents=True)
  (self.source/'base.usda').write_text('#usda 1.0\n(\n defaultPrim = "Root"\n metersPerUnit = 0.01\n upAxis = "Z"\n subLayers = [@geometry.usda@]\n)\n')
  (self.source/'geometry.usda').write_text('#usda 1.0\ndef Xform "Root" {}\n')
  self.contract={'schema':'seasonal-source-contract-v1','root_transform_identity':True,'stage_meters_per_unit':.01,'stage_up_axis':'Z','base_scene':'base.usda','base_sha256':t.sha(self.source/'base.usda'),'source_pins':{p.name:t.sha(p) for p in self.source.iterdir()},'anchors':{'table':{'kind':'horizontal','origin_cm':[200,100,80],'yaw_deg':90,'surface_z':0,'rectangle':[-50,-50,50,50],'blocked':[],'source_prim':'/Root/Table','method':'synthetic plane'},'mount':{'kind':'mount','origin_cm':[300,100,0],'yaw_deg':0,'source_prim':'/Root/Frame','method':'synthetic support posts','mount_points_local_cm':[[-59,0,168],[59,0,168]]}},'protected_bounds_cm':{'robot':{'min':[-30,-30,0],'max':[30,30,100]}},'bench_existing_bounds':[]}
  t.save(self.config/'anchor-contract.json',self.contract)
  t.save(self.config/'light-override-allowlist.json',{'paths':{'/Root/Moon':{'type':'DistantLight','attributes':['inputs:intensity','inputs:color']}}})
  (self.assets/'toy.usda').write_text('#usda 1.0\n( defaultPrim = "Asset"\n metersPerUnit = 1\n upAxis = "Z"\n)\ndef Xform "Asset" {}\n')
  (self.assets/'atlas.txt').write_text('synthetic dependency')
  self.asset={'id':'toy','path':'toy.usda','sha256':t.sha(self.assets/'toy.usda'),'meters_per_unit':1,'up_axis':'Z','default_prim':'/Asset','physics_present':False,'local_bounds':{'min':[-.1,-.2,.05],'max':[.1,.2,.45]},'dependencies':[{'path':'atlas.txt','sha256':t.sha(self.assets/'atlas.txt')}]}
  t.save(self.assets/'catalog.json',{'schema':'seasonal-imported-assets-r1','assets':[self.asset]})
  self.profile={'id':'halloween','status':'source_candidate','title':'Halloween','audio':{'soundtrack':'external-slot'},'placements':[{'id':'Toy','asset_id':'toy','anchor':'table','xy':[0,0],'height_cm':10,'yaw_deg':30}],'decorations':[{'id':'Bats','kind':'bat_garland','anchor':'mount'}],'new_lights':[],'existing_light_overrides':{'/Root/Moon':{'type':'DistantLight','values':{'inputs:intensity':30,'inputs:color':[.3,.4,1]}}},'asset_catalogs':['catalog.json']}
  self.profile_save(self.profile)
 def profile_save(self,p):t.save(self.config/'profiles'/f"{p['id']}.json",p)
 def build(self,season='halloween'):
  return t.build(season,config_root=self.config,source_root=self.source,asset_root=self.assets,output=self.root/season)
 def test_seating_rotation_height(self):
  r=self.build()['placements'][0]
  self.assertAlmostEqual(r['support_gap_cm'],0);self.assertEqual(r['source_center_minz'],[0,0,.05]);self.assertAlmostEqual(r['scale_source_metres_to_stage_cm'],25)
  self.assertAlmostEqual(r['world_bounds_cm']['min'][2],80);self.assertAlmostEqual(r['world_bounds_cm']['max'][2],90)
  self.assertNotIn('Physics',(self.root/'halloween/theme.usda').read_text())
 def test_three_season_replace_and_source_preservation(self):
  originals={p.name:p.read_bytes() for p in self.source.iterdir()};self.build()
  for season in ('christmas','neutral'):
   p=copy.deepcopy(self.profile);p.update(id=season,title=season,audio=None,placements=[],asset_catalogs=[],existing_light_overrides={})
   p['decorations']=[{'id':'Holiday','kind':'holiday_garland','anchor':'mount'}] if season=='christmas' else []
   p['status']='starter_not_native_reviewed' if season=='christmas' else 'source_candidate';self.profile_save(p);self.build(season)
  for season in ('halloween','christmas','neutral'):
   r=t.select(self.root/season,source_root=self.source,selection_root=self.root/'selected');usd=(self.root/'selected/selected.usda').read_text()
   self.assertIn('../'+season+'/theme.usda',usd);self.assertEqual(len(re.findall(r'@[^@]+@',usd)),2)
   for other in ('halloween','christmas','neutral'):
    if other!=season:self.assertNotIn('/'+other+'/',usd)
   self.assertEqual(r['theme'],season)
  self.assertIsNone(r['audio']);self.assertNotIn('SphereLight',(self.root/'neutral/theme.usda').read_text());self.assertNotIn('Bats',(self.root/'christmas/theme.usda').read_text())
  self.assertEqual(originals,{p.name:p.read_bytes() for p in self.source.iterdir()})
 def test_neutral_has_no_asset_dependency(self):
  p=copy.deepcopy(self.profile);p.update(id='neutral',placements=[],decorations=[],asset_catalogs=[],existing_light_overrides={},audio=None);self.profile_save(p)
  for path in self.assets.iterdir():path.unlink()
  self.assertEqual(self.build('neutral')['asset_dependencies'],{})
 def test_source_and_dependency_corruption(self):
  (self.assets/'atlas.txt').write_text('changed')
  with self.assertRaisesRegex(ValueError,'dependency'):self.build()
  (self.source/'geometry.usda').write_text('changed')
  with self.assertRaisesRegex(ValueError,'Protected source'):self.build()
 def test_duplicate_catalog_escape_and_bad_interface(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):t.asset_catalog(['catalog.json']*2,self.assets)
  for path in ('../source/base.usda',str(self.source/'base.usda')):
   with self.assertRaises(ValueError):t.contained(self.assets,path)
  (self.assets/'escape').symlink_to(self.source)
  with self.assertRaisesRegex(ValueError,'symlink'):t.contained(self.assets,'escape/base.usda')
  a=copy.deepcopy(self.asset);a['physics_present']=True;t.save(self.assets/'bad.json',{'schema':'seasonal-imported-assets-r1','assets':[a]})
  with self.assertRaisesRegex(ValueError,'interface'):t.asset_catalog(['bad.json'],self.assets)
 def test_support_and_protected_zone(self):
  p=copy.deepcopy(self.profile);p['placements'][0]['xy']=[49,0];self.profile_save(p)
  with self.assertRaisesRegex(ValueError,'support patch'):self.build()
  self.profile_save(self.profile);self.contract['protected_bounds_cm']['robot']={'min':[190,90,70],'max':[210,110,100]};t.save(self.config/'anchor-contract.json',self.contract)
  with self.assertRaisesRegex(ValueError,'Protected zone'):self.build()
 def test_existing_output_and_selection_protected(self):
  self.build()
  with self.assertRaises(FileExistsError):self.build()
  with self.assertRaisesRegex(ValueError,'separate'):t.build('halloween',config_root=self.config,source_root=self.source,asset_root=self.assets,output=self.source/'bad')
  selected=self.root/'selected';selected.mkdir();(selected/'important.txt').write_text('keep')
  with self.assertRaisesRegex(ValueError,'unowned'):t.select(self.root/'halloween',source_root=self.source,selection_root=selected)
  self.assertEqual((selected/'important.txt').read_text(),'keep')
 def test_selection_rejects_modified_build(self):
  self.build();(self.root/'halloween/theme.usda').write_text('changed')
  with self.assertRaisesRegex(ValueError,'Built layer'):t.select(self.root/'halloween',source_root=self.source,selection_root=self.root/'selected')
 def test_light_scope_duplicate_and_units(self):
  for path,typ in (('/Root/Robot','Xform'),('/Root/Moon','SphereLight')):
   p=copy.deepcopy(self.profile);p['existing_light_overrides']={path:{'type':typ,'values':{'inputs:intensity':30,'inputs:color':[.3,.4,1]}}};self.profile_save(p)
   with self.assertRaisesRegex(ValueError,'allowlisted'):self.build()
  p=copy.deepcopy(self.profile);p['decorations'][0]['id']='Toy';self.profile_save(p)
  with self.assertRaisesRegex(ValueError,'Duplicate'):self.build()
  self.profile_save(self.profile);self.contract['stage_meters_per_unit']=1;t.save(self.config/'anchor-contract.json',self.contract)
  with self.assertRaisesRegex(ValueError,'centimetres'):self.build()
 def test_actual_tether_connectivity(self):
  for kind in ('bat_garland','holiday_garland','cloth_ghost','web'):
   usd,_,_=t.decoration({'id':'Decor','kind':kind},self.contract['anchors']['mount'])
   def vertices(n):
    body=re.search(r'"'+n+r'"[^\{]*\{(.*?)\n\}',usd,re.S).group(1)
    match=re.search(r'point3f\[\] points = \[(.*?)\]\n',body,re.S)
    return [tuple(map(float,p.split(','))) for p in re.findall(r'\(([^)]+)\)',match.group(1))]
   if kind=='bat_garland':
    for i in range(4):self.assertIn(vertices('Tether'+str(i))[-1],vertices('Bat'+str(i)))
   elif kind=='cloth_ghost':self.assertIn(vertices('Tether')[-1],vertices('Cloth'))
   elif kind=='web':
    spokes=[vertices('PlanarSpoke'+str(i)) for i in range(8)];self.assertTrue(all(v[1]==-37 for s in spokes for v in s));self.assertEqual(vertices('WebRing2')[:-1],[s[-1] for s in spokes])
 def test_actual_base_units_and_mounts_must_match(self):
  self.contract['anchors']['mount']['mount_points_local_cm'][0][2]=170;t.save(self.config/'anchor-contract.json',self.contract)
  with self.assertRaisesRegex(ValueError,'matching display mounts'):self.build()
  self.contract['anchors']['mount']['mount_points_local_cm'][0][2]=168
  base=self.source/'base.usda';base.write_text(base.read_text().replace('metersPerUnit = 0.01','metersPerUnit = 1'))
  self.contract['base_sha256']=t.sha(base);self.contract['source_pins']['base.usda']=t.sha(base);t.save(self.config/'anchor-contract.json',self.contract)
  with self.assertRaisesRegex(ValueError,'root metadata'):self.build()
 def test_packaged_profiles_external_and_christmas_starter(self):
  profiles=ROOT/'seasonal-profiles/profiles';c=json.loads((profiles/'christmas.json').read_text());h=json.loads((profiles/'halloween.json').read_text())
  self.assertIn('draft',c['status']);self.assertEqual(c['asset_catalogs'],[]);self.assertIsNone(c['audio']);self.assertIsNone(h['audio']['soundtrack']);self.assertEqual(len(h['placements']),4)
if __name__=='__main__':unittest.main()
