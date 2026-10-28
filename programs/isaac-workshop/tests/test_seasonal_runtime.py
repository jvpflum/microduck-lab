"""Synthetic runtime checks; no Kit or render claims."""
import copy
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from isaac_workshop import seasonal_runtime as r
class RuntimeTests(unittest.TestCase):
 def setUp(self):self.receipt={'light_animation':[{'path':'/Root/SeasonalTheme/Garland/Lamp','base':450,'amplitude':.08,'frequency_hz':.7,'phase':0}]}
 def test_explicit_seconds_bounded_deterministic(self):
  values=[next(iter(r.sample_light_values(self.receipt,i/25).values())) for i in range(300)]
  self.assertEqual(values[0],450);self.assertGreater(max(values)-min(values),70);self.assertTrue(all(414<=v<=486 for v in values));self.assertEqual(r.sample_light_values(self.receipt,1.234),r.sample_light_values(self.receipt,1.234))
 def test_bad_time_scope_duplicate_and_amplitude(self):
  for time in (-1,math.nan,math.inf,True):
   with self.assertRaises(ValueError):r.sample_light_values(self.receipt,time)
  for field,value in (('path','/Root/Robot/Lamp'),('amplitude',.16),('frequency_hz',-1)):
   a=copy.deepcopy(self.receipt);a['light_animation'][0][field]=value
   with self.assertRaises(ValueError):r.sample_light_values(a,0)
  a=copy.deepcopy(self.receipt);a['light_animation']*=2
  with self.assertRaises(ValueError):r.sample_light_values(a,0)
 def test_clock_owns_only_intensity_and_cleans_up(self):
  writes=[];session=SimpleNamespace(subLayerPaths=['prior']);layer=SimpleNamespace(identifier='owned')
  class Prim:
   def GetTypeName(self):return 'SphereLight'
   def CreateAttribute(self,key,typ):self.key,self.typ=key,typ;return self
   def Set(self,value):writes.append((self.path,self.key,self.typ,value))
  class Edit:
   def OverridePrim(self,path):p=Prim();p.path=path;return p
  stage=SimpleNamespace(GetSessionLayer=lambda:session,GetPrimAtPath=lambda p:Prim())
  pxr=SimpleNamespace(Sdf=SimpleNamespace(Layer=SimpleNamespace(CreateAnonymous=lambda n:layer),ValueTypeNames=SimpleNamespace(Float='float')),Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda p:Edit())))
  with patch.dict(sys.modules,{'pxr':pxr}):
   clock=r.ThemeClock(stage,self.receipt);self.assertEqual(session.subLayerPaths,['owned','prior'])
   result=clock.apply(1);self.assertFalse(result['timeline_advanced']);self.assertEqual(result['robot_properties_written'],0)
   self.assertEqual(writes[0][:3],('/Root/SeasonalTheme/Garland/Lamp','inputs:intensity','float'))
   session.subLayerPaths.append('other');self.assertTrue(clock.close()['owned_light_layer_removed']);self.assertEqual(session.subLayerPaths,['prior','other']);self.assertTrue(clock.close()['owned_light_layer_removed'])
   with self.assertRaises(RuntimeError):clock.apply(2)
 def test_wrong_actual_light_type_does_not_install(self):
  session=SimpleNamespace(subLayerPaths=['prior']);stage=SimpleNamespace(GetSessionLayer=lambda:session,GetPrimAtPath=lambda p:SimpleNamespace(GetTypeName=lambda:'Xform'))
  pxr=SimpleNamespace(Sdf=SimpleNamespace(Layer=SimpleNamespace(CreateAnonymous=lambda n:SimpleNamespace(identifier='owned'))),Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda p:None)))
  with patch.dict(sys.modules,{'pxr':pxr}):
   with self.assertRaises(ValueError):r.ThemeClock(stage,self.receipt)
  self.assertEqual(session.subLayerPaths,['prior'])
if __name__=='__main__':unittest.main()
