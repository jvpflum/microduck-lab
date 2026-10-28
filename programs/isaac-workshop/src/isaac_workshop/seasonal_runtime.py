"""Explicit movie-time light modulation; never advances a USD timeline or robot pose."""
import math

NAMESPACE = '/Root/SeasonalTheme/'

def sample_light_values(receipt, seconds):
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Expected finite nonnegative movie seconds')
    result = {}
    for light in receipt['light_animation']:
        path = light['path']
        if not path.startswith(NAMESPACE) or path in result:
            raise ValueError('Light must be unique and seasonal-owned')
        base, amplitude, frequency, phase = (light[x] for x in ('base','amplitude','frequency_hz','phase'))
        if not all(math.isfinite(x) for x in (base, amplitude, frequency, phase)) or base < 0 or not 0 <= amplitude <= .15 or frequency < 0:
            raise ValueError('Invalid subtle flicker parameters')
        result[path] = base * (1 + amplitude * math.sin(2*math.pi*frequency*seconds + phase))
    return result

class ThemeClock:
    """One disposable session sublayer, exclusively for this profile's light intensities.

    The renderer retains this object, calls apply(seconds) before each capture,
    calls close() before swapping themes. No authored base-layer edits occur.
    """
    def __init__(self, stage, receipt):
        from pxr import Sdf, Usd
        self.stage, self.receipt = stage, receipt
        self.layer = Sdf.Layer.CreateAnonymous('seasonal-light-clock.usda')
        self.edit_stage = Usd.Stage.Open(self.layer)
        self.session = stage.GetSessionLayer()
        self.closed = False
        self.paths = sample_light_values(receipt, 0)
        for path in self.paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or prim.GetTypeName() != 'SphereLight':
                raise ValueError('Expected actual seasonal SphereLight: '+path)
        self.session.subLayerPaths = [self.layer.identifier] + list(self.session.subLayerPaths)

    def apply(self, seconds):
        from pxr import Sdf
        if self.closed:
            raise RuntimeError('Theme clock closed')
        values = sample_light_values(self.receipt, seconds)
        for path, value in values.items():
            self.edit_stage.OverridePrim(path).CreateAttribute('inputs:intensity', Sdf.ValueTypeNames.Float).Set(value)
        return {'seconds': seconds, 'intensities':values,'timeline_advanced':False,'robot_properties_written':0}

    def close(self):
        if not self.closed:
            self.session.subLayerPaths = [p for p in self.session.subLayerPaths if p != self.layer.identifier]
            self.closed = True
        return {'owned_light_layer_removed':self.layer.identifier not in self.session.subLayerPaths}

def apply_theme_time(clock, seconds):
    return clock.apply(seconds)

def canonical_light_readback(stage, receipt):
    """Future native integration gate; records actual canonical UsdLux inputs.

    Matching source values alone is not evidence of effective rendered illumination.
    Run after Kit's light/schema initialization, before accepting a themed preview.
    """
    rows=[]
    for path, spec in receipt['existing_light_overrides'].items():
        prim=stage.GetPrimAtPath(path)
        if not prim or prim.GetTypeName()!=spec['type']:
            raise ValueError('Native light path/type differs: '+path)
        for key, expected in spec['values'].items():
            if not key.startswith('inputs:'): continue
            attr=prim.GetAttribute(key); value=attr.Get() if attr else None
            value=list(value) if isinstance(expected,list) and value is not None else value
            if value is None:
                raise ValueError('Canonical light input unavailable: '+path+'.'+key)
            if isinstance(expected,list) and len(value)!=len(expected): raise ValueError('Canonical input dimension')
            pairs=zip(value,expected) if isinstance(expected,list) else [(value,expected)]
            if any(not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6) for a,b in pairs):
                raise ValueError('Canonical light input mismatch: '+path+'.'+key)
            rows.append({'path':path,'attribute':key,'value':value,'expected':expected,'type':str(attr.GetTypeName())})
    return {'canonical_inputs_match':True,'rows':rows,'rendered_light_effectiveness_verified':False}
