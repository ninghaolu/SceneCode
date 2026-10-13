"""Explicit reference-based diagnostic plan, blockout, and evaluated-mesh checks.

Dimensions and hidden structure are documented assumptions, not image ground truth.
The deterministic blockout code is never supplied to the implementation model.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

GROUPS = ('base_frame', 'legs', 'armrests', 'seat_cushions', 'back_cushions')
ROOT = Path(__file__).resolve().parents[1]


def make_plan():
    parts = []

    def add(name, group, center, dimensions, bevel=0, shape='rounded_box', material='fabric'):
        parts.append(dict(name=name, group=group, center=center, dimensions=dimensions,
                          bevel=bevel, shape=shape, material=material))

    add('base', 'base_frame', [0, 0, .20], [1.68, .82, .16], .028)
    add('back_support', 'base_frame', [0, .355, .485], [1.68, .11, .41], .018)
    for x, side in [(-.84, 'left'), (.84, 'right')]:
        for y, end in [(-.33, 'front'), (.33, 'rear')]:
            add(f'{side}_{end}_foot', 'legs', [x, y, .0625], [.068, .068, .125],
                shape='tapered_foot', material='wood')
    for x, side in [(-.90, 'left'), (.90, 'right')]:
        add(f'{side}_arm', 'armrests', [x, 0, .40], [.12, .84, .56], .04)
    for x, side in [(-.555, 'left'), (0, 'middle'), (.555, 'right')]:
        add(f'{side}_seat', 'seat_cushions', [x, -.065, .38], [.548, .65, .20], .045)
        add(f'{side}_back', 'back_cushions', [x, .255, .67], [.548, .17, .46], .055)
    contacts = [['base', p['name']] for p in parts if p['group'] in ('legs', 'armrests', 'seat_cushions')]
    contacts.append(['base', 'back_support'])
    for side in ('left', 'middle', 'right'):
        contacts.extend([[f'{side}_back', f'{side}_seat'], [f'{side}_back', 'back_support']])
    return dict(
        kind='assisted_supplied_plan_diagnostic_v1', units='meters', up='+Z', front='-Y',
        overall_dimensions=[1.92, .84, .90], floor_z=0, parts=parts, contacts=contacts,
        geometry_tolerance_m=.008,
        observed=['Three adjacent seat cushions and three broad rectangular back cushions',
                  'Two padded side arms, continuous upholstered base, short wooden feet',
                  'Light gray upholstery; low wide silhouette and rounded cushion edges'],
        assumptions=['Absolute width 1.92 m, depth .84 m and height .90 m are chosen plausible dimensions',
                     'Single reference does not determine exact depth, unseen rear structure, or feet',
                     'Four straight tapered feet approximate the visible short wooden feet',
                     'Hidden rear support is added to support the back cushions',
                     'Axis-aligned cushions and simple fabric omit seams, wrinkles and exact cushion bulging',
                     'Intentional seat/back overlap (.04 m vertically, .09 m in depth) represents cushion contact; '
                     'rear support overlaps backs by .04 m; feet overlap base by .005 m'],
        acceptance='Coarse reference structure and plausible support; not photorealistic or exact reconstruction. '
                   'Every evaluated component must match assigned bounds, counts and support contacts, '
                   'followed by independent three-view review. AABB tests alone cannot certify surface quality.')


def check_geometry(plan, measurements):
    errors, matched = [], {}
    tol = plan['geometry_tolerance_m']
    for group in GROUPS:
        expected = [p for p in plan['parts'] if p['group'] == group]
        actual = measurements.get(group, {}).get('components', [])
        if len(actual) != len(expected):
            errors.append(f'{group}: expected {len(expected)} connected components, found {len(actual)}')
            continue
        def distance(pair):
            return max(abs(pair[0][field][i]-pair[1][field][i])
                       for field in ('center', 'dimensions') for i in range(3))
        assignment = min(itertools.permutations(actual),
                         key=lambda candidate: sum(distance(pair) for pair in zip(expected, candidate)))
        for part, component in zip(expected, assignment):
            matched[part['name']] = component
            deviation = distance((part, component))
            if deviation > tol:
                errors.append(f"{part['name']}: maximum center/dimension deviation {deviation:.5f} m > {tol}")
    for first, second in plan['contacts']:
        if first not in matched or second not in matched:
            continue
        a, b = matched[first], matched[second]
        gap = max(max(a['min'][i]-b['max'][i], b['min'][i]-a['max'][i]) for i in range(3))
        if gap > tol:
            errors.append(f'{first}/{second}: separated bounds, gap {gap:.5f} m')
    if len(matched) == len(plan['parts']):
        lo = [min(c['min'][i] for c in matched.values()) for i in range(3)]
        hi = [max(c['max'][i] for c in matched.values()) for i in range(3)]
        if abs(lo[2]-plan['floor_z']) > tol:
            errors.append('Feet do not meet declared floor')
        if any(abs(hi[i]-lo[i]-plan['overall_dimensions'][i]) > tol for i in range(3)):
            errors.append('Overall dimensions disagree with frozen plan')
    return dict(passed=not errors, errors=errors, matched_parts=len(matched),
                scope='Evaluated mesh connected components, bounds and AABB support; visual review still required')


BLOCKOUT = '''import bpy
PARTS = {parts!r}
for p in PARTS:
    if p['shape'] == 'tapered_foot':
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=.025, radius2=.034,
                                       depth=p['dimensions'][2], location=p['center'])
    else:
        bpy.ops.mesh.primitive_cube_add(size=1, location=p['center'])
        bpy.context.object.dimensions = p['dimensions']
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj = bpy.context.object
    obj.name = p['name']
    if p['bevel']:
        mod = obj.modifiers.new('rounded edges', 'BEVEL')
        mod.width, mod.segments = p['bevel'], 5
        bpy.ops.object.modifier_apply(modifier=mod.name)
        normal = obj.modifiers.new('weighted normals', 'WEIGHTED_NORMAL')
        bpy.ops.object.modifier_apply(modifier=normal.name)
    mat = bpy.data.materials.get(p['material']) or bpy.data.materials.new(p['material'])
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.62,.63,.65,1) if p['material']=='fabric' else (.35,.19,.085,1)
    shader.inputs['Roughness'].default_value = .9 if p['material']=='fabric' else .5
    obj.data.materials.append(mat)
'''


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = make_plan()
    reference = next(p['path'] for p in json.loads((ROOT/'slurm/qwen_saved_images.json').read_text()) if 'sofa' in p['aliases'])
    reference_bytes = Path(reference).read_bytes()
    (output/'reference.png').write_bytes(reference_bytes)
    plan.update(reference_source=reference, reference_sha256=hashlib.sha256(reference_bytes).hexdigest())
    (output/'plan.json').write_text(json.dumps(plan, indent=2))
    code = output/'blockout/code'
    code.mkdir(parents=True)
    for group in GROUPS:
        (code/f'{group}.py').write_text(BLOCKOUT.format(parts=[p for p in plan['parts'] if p['group']==group]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, required=True)
    prepare(parser.parse_args().prepare)
