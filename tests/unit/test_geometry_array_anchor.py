"""An array's centroid cannot stand in for its declared template component."""
from types import SimpleNamespace

import pytest

from scenecode.code_object.agent_framework.utils.geometry_guard import part_position_guard


@pytest.mark.parametrize('correct', [False, True])
def test_array_requires_actual_template_component(correct):
    bpy = pytest.importorskip('bpy')
    before = set(bpy.data.objects)
    plan = {
        'name': 'array_anchor_guard', 'position': {'x': -.42, 'y': -.15, 'z': .025},
        'rotation': {'x': 0, 'y': 0, 'z': 0},
        'is_symmetric': True, 'symmetric_axis': 'x',
        'shape': {'base_shape': 'cylinder',
                  'dimensions': {'width': .035, 'depth': .035, 'height': .05},
                  'modifiers': ['array_rotate_z_4']},
    }
    positions = ([(-.42, -.15), (.42, -.15), (-.42, .15), (.42, .15)] if correct
                 else [(.42, 0), (-.15, .27), (-.42, -.30), (.15, -.57)])
    try:
        objects = []
        for x, y in positions:
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=.0175, depth=.05,
                                               location=(x, y, .025))
            objects.append(bpy.context.object)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
        bpy.context.object.name = plan['name']
        guard = part_position_guard(SimpleNamespace(to_dict=lambda: plan))
        if correct:
            exec(guard, {})
        else:
            with pytest.raises(ValueError, match='Geometry placement mismatch'):
                exec(guard, {})
    finally:
        for obj in set(bpy.data.objects) - before:
            bpy.data.objects.remove(obj, do_unlink=True)
