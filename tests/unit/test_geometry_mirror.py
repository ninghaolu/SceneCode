"""A declared mirror must keep both copies through constructor repair."""
from types import SimpleNamespace

import pytest

from scenecode.code_object.agent_framework.utils.geometry_guard import part_position_guard


@pytest.mark.parametrize('declaration', ['modifier', 'flag'])
@pytest.mark.parametrize('representation', ['joined', 'parent', 'modifier', 'missing'])
def test_mirrored_pair_coverage(declaration, representation):
    bpy = pytest.importorskip('bpy')
    before = set(bpy.data.objects)
    plan = {
        'name': 'mirror_guard', 'position': {'x': -.46, 'y': 0, 'z': .23},
        'rotation': {'x': 0, 'y': 0, 'z': 0},
        'is_symmetric': declaration == 'flag', 'symmetric_axis': 'x',
        'shape': {'base_shape': 'cube',
                  'dimensions': {'width': .08, 'depth': .38, 'height': .22},
                  'modifiers': ['mirror_x'] if declaration == 'modifier' else []},
    }
    try:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-.46, 0, .23))
        left = bpy.context.object
        left.name = 'mirror_guard'
        left.scale = (.08, .38, .22)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        if representation == 'modifier':
            origin = bpy.data.objects.new('mirror_origin', None)
            bpy.context.collection.objects.link(origin)
            mirror = left.modifiers.new('Mirror', 'MIRROR')
            mirror.mirror_object = origin
        elif representation != 'missing':
            right = left.copy()
            right.data = left.data.copy()
            bpy.context.collection.objects.link(right)
            right.location.x = .46
            if representation == 'joined':
                right.select_set(True)
                bpy.context.view_layer.objects.active = left
                bpy.ops.object.join()
            else:
                left.name = 'mirror_left'
                parent = bpy.data.objects.new('mirror_guard', None)
                bpy.context.collection.objects.link(parent)
                left.parent = right.parent = parent
        guard = part_position_guard(SimpleNamespace(to_dict=lambda: plan))
        if representation == 'missing':
            with pytest.raises(ValueError, match='mirror coverage mismatch'):
                exec(guard, {})
        else:
            exec(guard, {})
    finally:
        for obj in set(bpy.data.objects) - before:
            bpy.data.objects.remove(obj, do_unlink=True)
