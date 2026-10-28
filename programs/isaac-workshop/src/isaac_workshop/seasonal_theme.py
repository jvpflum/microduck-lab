#!/usr/bin/env python3
"""Source-preserving seasonal USD overlays; no simulator or rendering imports."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re

NAMESPACE = '/Root/SeasonalTheme'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError('Unsafe USD identifier')
    return value

def num(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError('Nonfinite or boolean numeric value')
    return format(float(value), '.15g')

def vec(value):
    return '(' + ', '.join(num(x) for x in value) + ')'

def rel(path, directory):
    value = os.path.relpath(Path(path).resolve(), Path(directory).resolve())
    if any(x in value for x in '@\n\r'):
        raise ValueError('Unsafe asset path')
    return value.replace(os.sep, '/')

def bounds(points):
    return {'min': [min(p[i] for p in points) for i in range(3)],
            'max': [max(p[i] for p in points) for i in range(3)]}

def corners(box):
    return list(itertools.product(*zip(box['min'], box['max'])))

def overlap(a, b, tol=1e-7):
    return all(a['min'][i] < b['max'][i]-tol and a['max'][i] > b['min'][i]+tol for i in range(3))

def rotate(point, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [c*point[0]-s*point[1], s*point[0]+c*point[1], point[2]]

def world(point, anchor):
    return [a+b for a, b in zip(rotate(point, anchor['yaw_deg']), anchor['origin_cm'])]

def contained(root, relative):
    """Resolve a declared relative input without absolute paths or symlink escape."""
    if not isinstance(relative, str) or not relative or any(c in relative for c in '@\n\r\\'):
        raise ValueError('Unsafe relative input path')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Input must be root-contained')
    resolved = (Path(root).resolve() / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError('Input symlink escapes root')
    return resolved

def source_pins(contract, source_root):
    if contract['schema'] != 'seasonal-source-contract-v1':
        raise ValueError('Source contract schema')
    if contract['stage_up_axis'] != 'Z' or not math.isclose(contract['stage_meters_per_unit'], .01, abs_tol=1e-9):
        raise ValueError('Measured authoring requires Z-up centimetres')
    if contract.get('root_transform_identity') is not True:
        raise ValueError('Measured composed Root identity must be declared')
    pins = contract['source_pins']
    if not pins or pins.get(contract['base_scene']) != contract['base_sha256']:
        raise ValueError('Base must be included in declared source closure')
    for relative, expected in pins.items():
        if not re.fullmatch(r'[0-9a-f]{64}', expected) or sha(contained(source_root, relative)) != expected:
            raise ValueError('Protected source changed: ' + relative)
    base = contained(source_root, contract['base_scene'])
    header = base.read_text().split('\n)', 1)[0]
    units = re.search(r'\bmetersPerUnit\s*=\s*([0-9.eE+-]+)', header)
    axis = re.search(r'\bupAxis\s*=\s*"([^"]+)"', header)
    root = re.search(r'\bdefaultPrim\s*=\s*"([^"]+)"', header)
    if not units or not math.isclose(float(units[1]), contract['stage_meters_per_unit'], abs_tol=1e-9) or not axis or axis[1] != 'Z' or not root or root[1] != 'Root':
        raise ValueError('Base root metadata differs from source contract')
    return base

def asset_catalog(catalog_paths, asset_root):
    result = {}
    for relative in catalog_paths:
        catalog = load(contained(asset_root, relative))
        if catalog['schema'] != 'seasonal-imported-assets-r1':
            raise ValueError('Catalog schema')
        for a in catalog['assets']:
            name(a['id'])
            if a['id'] in result: raise ValueError('Duplicate imported asset ID')
            if a['meters_per_unit'] != 1 or a['up_axis'] != 'Z' or a['default_prim'] != '/Asset' or a['physics_present'] is not False:
                raise ValueError('Asset interface differs')
            b = a['local_bounds']
            if any(len(b[k]) != 3 for k in ('min', 'max')) or any(not math.isfinite(v) for k in b for v in b[k]) or any(b['min'][i] >= b['max'][i] for i in range(3)):
                raise ValueError('Invalid asset bounds')
            for item in [a] + a['dependencies']:
                path = contained(asset_root, item['path'])
                if sha(path) != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
                    raise ValueError('Asset dependency hash mismatch')
            result[a['id']] = a
    return result

def placement_record(p, asset, anchor):
    if anchor['kind'] != 'horizontal':
        raise ValueError('Imported prop needs a measured horizontal support')
    b = asset['local_bounds']
    scale = p['height_cm'] / (b['max'][2]-b['min'][2])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid target height')
    center = [(b['min'][i]+b['max'][i])/2 for i in (0, 1)] + [b['min'][2]]
    points = []
    for c in corners(b):
        q = rotate([(c[i]-center[i])*scale for i in range(3)], p['yaw_deg'])
        points.append([q[0]+p['xy'][0], q[1]+p['xy'][1], q[2]+anchor['surface_z']])
    local = bounds(points)
    r = anchor['rectangle']
    if not (r[0] <= local['min'][0] and r[1] <= local['min'][1] and local['max'][0] <= r[2] and local['max'][1] <= r[3]):
        raise ValueError('Prop exceeds support patch: ' + p['id'])
    if any(overlap(local, other) for other in anchor['blocked']):
        raise ValueError('Prop overlaps existing shelf dressing: ' + p['id'])
    return {'id': p['id'], 'asset_id': p['asset_id'], 'anchor': p['anchor'], 'source_sha256': asset['sha256'],
            'scale_source_metres_to_stage_cm': scale, 'source_center_minz': center,
            'anchor_local_bounds_cm': local, 'world_bounds_cm': bounds([world(q, anchor) for q in points]),
            'support_gap_cm': local['min'][2]-anchor['surface_z']}

def xform_start(n, translation=(0,0,0), yaw=0, scale=None):
    text = f'def Xform "{name(n)}" {{\n double3 xformOp:translate = {vec(translation)}\n double xformOp:rotateZ = {num(yaw)}\n'
    order = ['xformOp:translate', 'xformOp:rotateZ']
    if scale is not None:
        text += f' double3 xformOp:scale = {vec([scale]*3)}\n'
        order.append('xformOp:scale')
    return text + ' uniform token[] xformOpOrder = ' + json.dumps(order) + '\n'

def material(n, color, emission=(0,0,0)):
    return f'''def Material "{n}" {{
 token outputs:surface.connect = <{NAMESPACE}/Looks/{n}/Surface.outputs:surface>
 def Shader "Surface" {{
 uniform token info:id = "UsdPreviewSurface"
 color3f inputs:diffuseColor = {vec(color)}
 color3f inputs:emissiveColor = {vec(emission)}
 float inputs:roughness = 0.55
 float inputs:metallic = 0
 token outputs:surface
 }}
}}\n'''

def bind(mat):
    return f'rel material:binding = <{NAMESPACE}/Looks/{mat}>\n'

def sphere(n, center, radius, mat):
    return f'def Sphere "{name(n)}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{\n double radius = {num(radius)}\n double3 xformOp:translate = {vec(center)}\n uniform token[] xformOpOrder = ["xformOp:translate"]\n' + bind(mat) + '}\n'

def curve(n, points, width, mat):
    return f'''def BasisCurves "{name(n)}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
 uniform token type = "linear"
 uniform token wrap = "nonperiodic"
 int[] curveVertexCounts = [{len(points)}]
 point3f[] points = [{', '.join(vec(p) for p in points)}]
 float[] widths = [{num(width)}] (interpolation = "constant")
 {bind(mat)}}}\n'''

def mesh(n, points, faces, mat):
    return f'''def Mesh "{name(n)}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
 point3f[] points = [{', '.join(vec(p) for p in points)}]
 int[] faceVertexCounts = {json.dumps([len(f) for f in faces])}
 int[] faceVertexIndices = {json.dumps([v for f in faces for v in f])}
 uniform token subdivisionScheme = "none"
 bool doubleSided = true
 {bind(mat)}}}\n'''

def decoration(d, anchor):
    """Small supported decorative models. All coordinates are measured display-local cm."""
    text = xform_start(d['id'], anchor['origin_cm'], anchor['yaw_deg'])
    kind = d['kind']; points = []; lights = []
    def add_curve(n, p, width=.4, mat='Cord'):
        nonlocal text
        text += curve(n, p, width, mat); points.extend(p)
    if kind in ('bat_garland', 'holiday_garland'):
        cord = [[-59,0,168],[-59,-36,165]] + [[x,-36,159+6*(x/59)**2] for x in range(-59,60,10)] + [[59,-36,165],[59,0,168]]
        add_curve('SupportedCord', cord)
        if kind == 'bat_garland':
            silhouette = [(-12,4),(-7,2),(-4,5),(-2,1),(0,3),(2,1),(4,5),(7,2),(12,4),(8,-4),(5,-2),(2,-5),(0,-3),(-2,-5),(-5,-2),(-8,-4)]
            for i,x in enumerate((-43,-15,15,43)):
                z=159+6*(x/59)**2
                p=[[x+u,-36,z-9+v] for u,v in silhouette]
                add_curve(f'Tether{i}',[[x,-36,z],[x,-36,z-6]],.25)
                text += mesh(f'Bat{i}',p,[list(range(len(p)))],'Violet'); points.extend(p)
        else:
            for i,x in enumerate(range(-50,51,20)):
                z=159+6*(x/59)**2
                add_curve(f'Tether{i}',[[x,-36,z],[x,-36,z-5]],.25)
                text += sphere(f'Bauble{i}',[x,-36,z-8],3,'Red' if i%2 else 'Green')
                points.extend([[x-3,-39,z-11],[x+3,-33,z-5]])
        for i,x in enumerate(range(-50,51,10)):
            z=159+6*(x/59)**2
            text += sphere(f'Bulb{i}',[x,-36,z],.7,'WarmBulb')
            text += f'def SphereLight "Practical{i}" {{\n double3 xformOp:translate = {vec([x,-36,z])}\n uniform token[] xformOpOrder = ["xformOp:translate"]\n float inputs:radius = 0.7\n float inputs:intensity = 450\n color3f inputs:color = (1, 0.44, 0.12)\n bool inputs:normalize = false\n}}\n'
            lights.append({'path': f'{NAMESPACE}/{d["id"]}/Practical{i}', 'base':450.0,'amplitude':.08,'frequency_hz':.7,'phase':i*.8})
    elif kind == 'cloth_ghost':
        add_curve('Tether',[[-59,0,168],[-59,-43,140]],.3)
        ring=[]
        for j,(z,r) in enumerate(((140,1),(137,5),(132,8),(124,9),(115,12))):
            for i in range(16):
                a=2*math.pi*i/16
                ring.append([-59+r*math.cos(a),-43+r*math.sin(a),z+(1.5*math.cos(4*a) if j==4 else 0)])
        faces=[[j*16+i,j*16+(i+1)%16,(j+1)*16+(i+1)%16,(j+1)*16+i] for j in range(4) for i in range(16)]
        ring.append([-59,-43,140]); faces += [[80,i,(i+1)%16] for i in range(16)]
        text += mesh('Cloth',ring,faces,'Cloth'); points.extend(ring)
        # Eyes face the room, source display's -Y direction.
        for i,x in enumerate((-62,-56)):
            text += sphere(f'Eye{i}',[x,-50.6,132],1.2,'Cord')
    elif kind == 'web':
        center=[0,-37,141]
        rims=[[24*math.cos(k*math.pi/4),-37,141+24*.6*math.sin(k*math.pi/4)] for k in range(8)]
        for i,p in enumerate(rims): add_curve(f'PlanarSpoke{i}',[center,p],.15,'Cloth')
        for i,k in enumerate((1,3,5,7)):
            p=rims[k]; support=[59 if p[0]>0 else -59,0,168 if p[2]>141 else 100]
            add_curve(f'MountTether{i}',[p,support],.15,'Cloth')
        for i,r in enumerate((8,16,24)):
            p=[[r*math.cos(k*math.pi/4),-37,141+r*.6*math.sin(k*math.pi/4)] for k in range(9)]
            add_curve(f'WebRing{i}',p,.12,'Cloth')
    else: raise ValueError('Unknown decoration kind')
    # Widen by sphere/cord radii; conservative and independent of robot geometry.
    b=bounds([world(p,anchor) for p in points]); b['min']=[x-1 for x in b['min']]; b['max']=[x+1 for x in b['max']]
    return text+'}\n', {'id':d['id'],'world_bounds_cm':b,'support':anchor['source_prim'],'support_method':anchor['method']}, lights

def tree_overrides(overrides):
    tree={}
    for path, spec in overrides.items():
        node=tree
        for part in path.strip('/').split('/'): node=node.setdefault(part,{})
        node['_values']=spec['values']
    def emit(nodes):
        text=''
        for n,v in nodes.items():
            if n=='_values': continue
            text+=f'over "{name(n)}" {{\n'
            for key,value in v.get('_values',{}).items():
                typ='color3f' if isinstance(value,list) else 'float'
                text+=f'{typ} {key} = '+(vec(value) if isinstance(value,list) else num(value))+'\n'
            text+=emit(v)+'}\n'
        return text
    return emit(tree.get('Root', {}))

def wrapper_text(base, theme, destination):
    original=base.read_text()
    metadata=original[:original.index('\n)')+2]
    metadata,n=re.subn(r'subLayers\s*=\s*\[[\s\S]*?\]',f'subLayers = [@{rel(theme,destination.parent)}@, @{rel(base,destination.parent)}@]',metadata,count=1)
    if n!=1: raise ValueError('Base root metadata has no unambiguous subLayers')
    return metadata+'\n'

def build(theme, *, config_root, source_root, asset_root, output):
    """Author a fresh overlay from supplied, measured contracts; never render or simulate.

    config_root contains profiles/<theme>.json, anchor-contract.json and
    light-override-allowlist.json. Output must be fresh and outside all input roots.
    Contracts attest a supplied source closure; this is not an asset resolver scan.
    """
    name(theme)
    config_root, source_root, asset_root = (Path(x).resolve() for x in (config_root, source_root, asset_root))
    profile_path = contained(config_root, f'profiles/{theme}.json')
    contract_path = contained(config_root, 'anchor-contract.json')
    profile=load(profile_path); contract=load(contract_path)
    if profile['id']!=theme: raise ValueError('Theme identity mismatch')
    base = source_pins(contract, source_root)
    allow_path = contained(config_root, 'light-override-allowlist.json')
    allow=load(allow_path)['paths']
    ids = [name(x['id']) for key in ('placements', 'decorations', 'new_lights') for x in profile[key]]
    if len(ids) != len(set(ids)) or 'Looks' in ids:
        raise ValueError('Duplicate or reserved theme prim ID')
    for path,override in profile['existing_light_overrides'].items():
        if not path.startswith('/Root/') or path.startswith(NAMESPACE) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) for part in path.strip('/').split('/')) or path not in allow or override['type'] not in ('RectLight', 'SphereLight', 'DomeLight', 'DistantLight', 'DiskLight', 'CylinderLight') or override['type']!=allow[path]['type'] or set(override['values'])!=set(allow[path]['attributes']):
            raise ValueError('Non-allowlisted existing light override')
    for override in profile['existing_light_overrides'].values():
        for attribute in override['values']:
            if not re.fullmatch(r'(?:inputs:)?(?:intensity|exposure|color|colorTemperature)', attribute):
                raise ValueError('Only explicit lighting value attributes may be overridden')
    out=Path(output).resolve()
    if any(out.is_relative_to(root) or root.is_relative_to(out) for root in (config_root,source_root,asset_root)):
        raise ValueError('Output must be separate from input roots')
    if out.exists(): raise FileExistsError('Use a fresh build directory')
    catalog=asset_catalog(profile['asset_catalogs'], asset_root); records=[]; lights=[]
    text='#usda 1.0\n(\n defaultPrim = "Root"\n metersPerUnit = '+num(contract['stage_meters_per_unit'])+'\n upAxis = "Z"\n)\nover "Root" {\ndef Xform "SeasonalTheme" {\n custom string theme:id = '+json.dumps(theme)+'\n'
    if profile['placements'] or profile['decorations']:
        text+='def Scope "Looks" {\n'+''.join(material(*x) for x in [('Cord',(.015,.012,.02)),('Violet',(.12,.025,.22)),('Cloth',(.88,.9,.84)),('Red',(.6,.025,.025)),('Green',(.025,.24,.09)),('WarmBulb',(1,.55,.12),(4,1.2,.1))])+'}\n'
    for p in profile['placements']:
        name(p['id']); a=catalog[p['asset_id']]; anchor=contract['anchors'][p['anchor']]
        r=placement_record(p,a,anchor); records.append(r)
        pos=world([*p['xy'],anchor['surface_z']],anchor)
        text+=xform_start(p['id'],pos,anchor['yaw_deg']+p['yaw_deg'],r['scale_source_metres_to_stage_cm'])
        text+=xform_start('Center',[-v for v in r['source_center_minz']])
        text+=f'def Xform "Source" (prepend references = @{rel(contained(asset_root,a["path"]),out)}@</Asset>) {{}}\n}}\n}}\n'
    for d in profile['decorations']:
        anchor=contract['anchors'][d['anchor']]
        if anchor['kind'] != 'mount' or anchor.get('mount_points_local_cm') != [[-59,0,168],[59,0,168]]:
            raise ValueError('Decoration recipe needs declared matching display mounts')
        t,r,ls=decoration(d,anchor); text+=t;records.append(r);lights+=ls
    for light in profile['new_lights']:
        if light['kind']!='rect' or light['fixture'] is not False: raise ValueError('Unsupported light')
        eye=light['eye_cm']; target=light['target_cm']
        z=[a-b for a,b in zip(eye,target)]; norm=math.sqrt(sum(x*x for x in z)); z=[x/norm for x in z]
        x=[-z[1],z[0],0]; norm=math.hypot(x[0],x[1]); x=[v/norm for v in x]
        y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]]
        matrix='('+', '.join(vec(row) for row in [x+[0],y+[0],z+[0],eye+[1]])+')'
        text+=f'def RectLight "{name(light["id"])}" {{\n matrix4d xformOp:transform = {matrix}\n uniform token[] xformOpOrder = ["xformOp:transform"]\n float inputs:width = {num(light["width_cm"])}\n float inputs:height = {num(light["height_cm"])}\n float inputs:intensity = {num(light["intensity"])}\n color3f inputs:color = {vec(light["color"])}\n bool inputs:normalize = false\n}}\n'
    text+='}\n'+tree_overrides(profile['existing_light_overrides'])+'}\n'
    for r in records:
        for label,box in contract['protected_bounds_cm'].items():
            if overlap(r['world_bounds_cm'],box): raise ValueError('Protected zone intrusion '+r['id']+' '+label)
        if r.get('anchor')=='bench_back' and any(overlap(r['world_bounds_cm'],b) for b in contract['bench_existing_bounds']):
            raise ValueError('Bench existing-object overlap '+r['id'])
    for a,b in itertools.combinations([r for r in records if 'asset_id' in r],2):
        if overlap(a['world_bounds_cm'],b['world_bounds_cm']): raise ValueError('Imported props overlap')
    out.mkdir(parents=True,exist_ok=False)
    (out/'theme.usda').write_text(text)
    (out/'workshop.usda').write_text(wrapper_text(base,out/'theme.usda',out/'workshop.usda'))
    receipt={'schema':'portable-seasonal-build-v1','theme':theme,'status':profile['status'],'profile_sha256':sha(profile_path),
             'base_sha256':contract['base_sha256'],'source_hashes_unchanged':True,'protected_source_pins':contract['source_pins'],'theme_sha256':sha(out/'theme.usda'),'wrapper_sha256':sha(out/'workshop.usda'),
             'placements':records,'light_animation':lights,'existing_light_overrides':profile['existing_light_overrides'],'title':profile['title'],'audio':profile['audio'],
             'native_preview_verified':False,'physics_or_motion_modified':False,'light_effectiveness_verified':False,
             'limits':['Source bounds/support and protected-zone checks only; no native appearance/physical-stability qualification.','New light values require actual canonical UsdLux readback and native preview.','Imported candle and pumpkin materials remain diffuse; separate visible bulb fixtures provide light.']}
    receipt['base_scene'] = contract['base_scene']
    receipt['input_sha256'] = {p.name:sha(p) for p in [Path(__file__), Path(__file__).with_name('seasonal_runtime.py'), contract_path, allow_path, profile_path]}
    receipt['catalog_sha256'] = {cp:sha(contained(asset_root, cp)) for cp in profile['asset_catalogs']}
    receipt['asset_dependencies'] = {item['path']:item['sha256'] for a in catalog.values() for item in [a]+a['dependencies']}
    # Media slots are metadata only: caller supplies and rights-checks any soundtrack.
    receipt['media_included'] = False
    source_pins(contract, source_root); save(out/'build-receipt.json',receipt)
    return receipt

def select(theme_directory, *, source_root, selection_root):
    """Persistently replace a selection with one overlay plus the unchanged base.

    Existing selection directories must contain our ownership marker. Close any
    ThemeClock and reopen selected.usda in the consumer when switching seasons.
    """
    directory = Path(theme_directory).resolve()
    receipt = load(directory/'build-receipt.json')
    if receipt['schema'] != 'portable-seasonal-build-v1': raise ValueError('Build schema')
    for filename, key in [('theme.usda', 'theme_sha256'), ('workshop.usda', 'wrapper_sha256')]:
        if sha(directory/filename) != receipt[key]: raise ValueError('Built layer changed')
    for relative, digest in receipt['protected_source_pins'].items():
        if sha(contained(source_root, relative)) != digest: raise ValueError('Source changed')
    base = contained(source_root, receipt['base_scene'])
    if sha(base) != receipt['base_sha256']: raise ValueError('Base changed')
    output = Path(selection_root).resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (directory, Path(source_root).resolve())):
        raise ValueError('Selection must be separate from source and build')
    marker = output/'.seasonal-selection-owner.json'
    if output.exists():
        if not marker.is_file() or load(marker) != {'schema':'seasonal-selection-owner-v1'}:
            raise ValueError('Refuse to replace unowned selection directory')
    else:
        output.mkdir(parents=True)
        save(marker, {'schema':'seasonal-selection-owner-v1'})
    for filename in ('selected.usda', 'selected.usda.tmp', 'selected-theme.json', 'selected-theme.json.tmp'):
        if (output/filename).is_symlink(): raise ValueError('Refuse selection symlink')
    temporary=output/'selected.usda.tmp'
    temporary.write_text(wrapper_text(base,directory/'theme.usda',output/'selected.usda'))
    os.replace(temporary, output/'selected.usda')
    selected={'schema':'portable-seasonal-selection-v1','theme':receipt['theme'],'status':receipt['status'],
              'selected_wrapper_sha256':sha(output/'selected.usda'),'build_receipt_sha256':sha(directory/'build-receipt.json'),
              'title':receipt['title'],'audio':receipt['audio'],'native_preview_verified':False}
    save(output/'selected-theme.json.tmp',selected)
    os.replace(output/'selected-theme.json.tmp',output/'selected-theme.json')
    return selected

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('theme')
    for option in ('config-root','source-root','asset-root','output'):
        parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--selection-root',type=Path)
    args=parser.parse_args()
    receipt=build(args.theme,config_root=args.config_root,source_root=args.source_root,asset_root=args.asset_root,output=args.output)
    if args.selection_root:
        select(args.output,source_root=args.source_root,selection_root=args.selection_root)
    print(json.dumps({'theme':receipt['theme'],'theme_sha256':receipt['theme_sha256'],'placements':len(receipt['placements']),'native_preview_verified':False}))

if __name__=='__main__':
    main()
