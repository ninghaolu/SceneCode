"""Blender-side checks that turn executable but misplaced parts into repair input."""
def part_position_guard(part):
    plan = part.to_dict()
    modifiers = plan["shape"].get("modifiers", [])
    mirror_axes = {str(m).lower().removeprefix("mirror_") for m in modifiers
                   if str(m).lower() in {"mirror_x", "mirror_y", "mirror_z"}}
    if plan.get("is_symmetric"):
        mirror_axes.update(a for a in plan.get("symmetric_axis", "x") if a in "xyz")
    # Explicit instances already enumerate every required object; do not mirror
    # those instances again.
    if plan.get("instances"):
        mirror_axes.clear()
    targets = plan.get("instances") or [{"name": plan["name"], "position": plan["position"]}]
    expected = [{"name": t["name"], "center": [t["position"][axis] for axis in "xyz"]}
                for t in targets]
    for target in expected:
        target["allowed_centers"] = [target["center"]]
        target["array_anchor"] = any(str(m).startswith("array") for m in modifiers)
        target["mirror_axes"] = sorted("xyz".index(a) for a in mirror_axes)
        target["dimensions"] = [plan["shape"]["dimensions"][axis]
                                for axis in ("width", "depth", "height")]
        target["rotation"] = [plan.get("rotation", {}).get(axis, 0.) for axis in "xyz"]
        # A cube's declared dimensions describe its finished envelope, not the
        # pre-modifier cage. Check named instances individually. A joined mirror
        # or array has group bounds, so retain its existing placement-only check.
        target["check_shrinkage"] = (
            plan["shape"].get("base_shape") == "cube"
            and not target["array_anchor"]
            and not mirror_axes
        )
        # A mirror may be exported either as separate objects or one joined mesh.
        # In the latter representation the group's center lies on the mirror plane.
        if mirror_axes and not target["array_anchor"]:
            mirrored = target["center"].copy()
            for axis in target["mirror_axes"]:
                mirrored[axis] = 0.
            target["allowed_centers"].append(mirrored)
    tolerance = max(.003, .03 * max(plan["shape"]["dimensions"].values()))
    return '''
import bpy as _guard_bpy
import math as _guard_math
from mathutils import Vector as _GuardVector
from mathutils import Euler as _GuardEuler
_guard_bpy.context.view_layer.update()
_guard_graph = _guard_bpy.context.evaluated_depsgraph_get()
for _guard_target in ''' + repr(expected) + ''':
    _guard_obj = _guard_bpy.data.objects.get(_guard_target['name'])
    if _guard_obj is None:
        raise ValueError('Geometry placement check: missing named part ' + _guard_target['name'])
    _guard_meshes = [_guard_obj] if _guard_obj.type == 'MESH' else [o for o in _guard_obj.children_recursive if o.type == 'MESH']
    _guard_points = []
    for _guard_mesh in _guard_meshes:
        _guard_eval = _guard_mesh.evaluated_get(_guard_graph)
        # Subdivision can retain cage-sized bound_box values. Measure the
        # evaluated vertices so unapplied modifiers cannot hide shrinkage.
        _guard_data = _guard_eval.to_mesh()
        try:
            _guard_points.extend(_guard_eval.matrix_world @ v.co for v in _guard_data.vertices)
        finally:
            _guard_eval.to_mesh_clear()
    if not _guard_points:
        raise ValueError('Geometry placement check: empty part ' + _guard_target['name'])
    _guard_min = [min(p[i] for p in _guard_points) for i in range(3)]
    _guard_max = [max(p[i] for p in _guard_points) for i in range(3)]
    _guard_center = [(_guard_min[i] + _guard_max[i]) / 2 for i in range(3)]
    if not all(_guard_math.isfinite(x) for x in _guard_min + _guard_max) or any(_guard_max[i] <= _guard_min[i] for i in range(3)):
        raise ValueError('Geometry placement check: invalid bounds for ' + _guard_target['name'])
    if _guard_target['mirror_axes']:
        _guard_rotation = _GuardEuler(tuple(_guard_math.radians(x) for x in _guard_target['rotation']), 'XYZ').to_matrix()
        for i in _guard_target['mirror_axes']:
            _guard_half_extent = sum(abs(_guard_rotation[i][j]) * _guard_target['dimensions'][j] / 2 for j in range(3))
            _guard_required = abs(_guard_target['center'][i]) + _guard_half_extent
            if (_guard_min[i] > -_guard_required + ''' + repr(tolerance) + ''' or
                    _guard_max[i] < _guard_required - ''' + repr(tolerance) + '''):
                raise ValueError('Geometry mirror coverage mismatch for ' + _guard_target['name'] +
                    ': declared mirrored copies must cover both sides of world axis ' + str(i) +
                    ', required extent=' + str(_guard_required) + ', measured bounds=' + str([_guard_min, _guard_max]) +
                    '. Preserve every declared mirror copy. Return the complete joined mesh or a named parent with all copies as children; '
                    'do not remove the mirror to satisfy a center check.')
    # Array position is the template's position, never the joined group center.
    # Accepting a matching group centroid can hide every misplaced copy.
    _guard_candidates = [] if _guard_target['array_anchor'] else [_guard_center]
    # A declared array is positioned by its template, not the whole group's
    # centroid. Check actual connected components for that template center.
    if _guard_target['array_anchor']:
        for _guard_mesh in _guard_meshes:
            _guard_eval = _guard_mesh.evaluated_get(_guard_graph)
            _guard_data = _guard_eval.to_mesh()
            _guard_adj = [set() for v in _guard_data.vertices]
            for _guard_edge in _guard_data.edges:
                a, b = _guard_edge.vertices
                _guard_adj[a].add(b)
                _guard_adj[b].add(a)
            _guard_unseen = set(range(len(_guard_adj)))
            while _guard_unseen:
                _guard_stack = [_guard_unseen.pop()]
                _guard_component = []
                while _guard_stack:
                    i = _guard_stack.pop()
                    _guard_component.append(_guard_eval.matrix_world @ _guard_data.vertices[i].co)
                    neighbours = _guard_adj[i] & _guard_unseen
                    _guard_unseen.difference_update(neighbours)
                    _guard_stack.extend(neighbours)
                if len(_guard_component) >= 4:
                    _guard_candidates.append([(min(p[i] for p in _guard_component) + max(p[i] for p in _guard_component))/2 for i in range(3)])
            _guard_eval.to_mesh_clear()
    if not _guard_candidates or min(max(abs(measured[i] - candidate[i]) for i in range(3)) for candidate in _guard_target['allowed_centers'] for measured in _guard_candidates) > ''' + repr(tolerance) + ''':
        raise ValueError('Geometry placement mismatch for ' + _guard_target['name'] +
            ': measured world bounds=' + str([_guard_min, _guard_max]) +
            ', measured center=' + str(_guard_center) + ', allowed PartPlan center(s)=' + str(_guard_target['allowed_centers']) +
            '. Fix the generated function, including its origin after joining meshes: bpy.ops.object.join retains the active panel origin. '
            'Center local geometry/origin before applying the final translation exactly once. Preserve world-space sub-part positions. '
            'For arrays, a real template component must occupy the declared position; do not move the entire group centroid there.')
    if _guard_target['check_shrinkage']:
        _guard_inverse_rotation = _GuardEuler(tuple(_guard_math.radians(x) for x in _guard_target['rotation']), 'XYZ').to_matrix().transposed()
        _guard_local_points = [_guard_inverse_rotation @ p for p in _guard_points]
        _guard_extents = [max(p[i] for p in _guard_local_points) - min(p[i] for p in _guard_local_points) for i in range(3)]
        # Allow the same positional tolerance on both faces of the envelope.
        if any(_guard_extents[i] < _guard_target['dimensions'][i] - ''' + repr(2 * tolerance) + ''' for i in range(3)):
            raise ValueError('Geometry dimension shrinkage for ' + _guard_target['name'] +
                ': evaluated extents in PartPlan axes=' + str(_guard_extents) +
                ', planned dimensions=' + str(_guard_target['dimensions']) +
                '. Preserve the declared finished dimensions and support faces after modifiers. '
                'Smooth shading does not require subdivision. Apply bevel/support geometry before any subdivision, '
                'in modifier stack order; do not turn a slab or rectangular cushion into a shrunken ellipsoid.')
'''
