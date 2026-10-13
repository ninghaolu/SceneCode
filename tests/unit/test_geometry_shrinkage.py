"""Check executed part envelopes rather than trusting pre-modifier dimensions."""
from types import SimpleNamespace
import math

import pytest

from scenecode.code_object.agent_framework.utils.geometry_guard import part_position_guard


@pytest.mark.parametrize('rotation', [0, 25])
@pytest.mark.parametrize('subdivision', [False, True])
def test_evaluated_cube_envelope_rejects_subdivision_shrinkage(rotation, subdivision):
    bpy = pytest.importorskip('bpy')
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = 'cushion_guard'
    obj.scale = (.30, .38, .12)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.location = (.31, -.02, .26)
    obj.rotation_euler.x = math.radians(rotation)
    if subdivision:
        mod = obj.modifiers.new('Subdivision', 'SUBSURF')
        mod.levels = 2
    else:
        mod = obj.modifiers.new('Bevel', 'BEVEL')
        mod.width = .015
        mod.segments = 4
    part = SimpleNamespace(to_dict=lambda: {
        'name': 'cushions', 'position': {'x': .31, 'y': -.02, 'z': .26},
        'rotation': {'x': rotation, 'y': 0, 'z': 0}, 'is_symmetric': True,
        'instances': [{'name': 'cushion_guard', 'position': {'x': .31, 'y': -.02, 'z': .26}}],
        'shape': {'base_shape': 'cube', 'dimensions': {'width': .30, 'depth': .38, 'height': .12},
                  'modifiers': ['bevel', 'smooth']},
    })
    try:
        if subdivision:
            with pytest.raises(ValueError, match='Geometry dimension shrinkage'):
                exec(part_position_guard(part), {})
        else:
            exec(part_position_guard(part), {})
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
