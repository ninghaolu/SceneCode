"""Standalone Blender CPU measurement/render harness for the saved sofa case.

Run with blender -b --python this_file -- --code PART_DIRECTORY --output DIR.
No model calls and no Drake/OpenGL renderer are used here.
"""
import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def measure(objects):
    graph = bpy.context.evaluated_depsgraph_get()
    components, all_points = [], []
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        points = [list(evaluated.matrix_world @ v.co) for v in mesh.vertices]
        all_points.extend(points)
        adjacency = [set() for _ in points]
        for edge in mesh.edges:
            a, b = edge.vertices
            adjacency[a].add(b)
            adjacency[b].add(a)
        remaining = set(range(len(points)))
        while remaining:
            stack, ids = [remaining.pop()], []
            while stack:
                index = stack.pop()
                ids.append(index)
                neighbors = adjacency[index] & remaining
                remaining.difference_update(neighbors)
                stack.extend(neighbors)
            if len(ids) >= 4:
                lo = [min(points[j][i] for j in ids) for i in range(3)]
                hi = [max(points[j][i] for j in ids) for i in range(3)]
                components.append({'object': obj.name, 'min': lo, 'max': hi,
                                   'dimensions': [hi[i]-lo[i] for i in range(3)],
                                   'center': [(hi[i]+lo[i])/2 for i in range(3)]})
        evaluated.to_mesh_clear()
    digest = hashlib.sha256(json.dumps(sorted(all_points)).encode()).hexdigest()
    return {'components': sorted(components, key=lambda c: c['center']),
            'geometry_sha256': digest, 'vertices': len(all_points)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measure-only', action='store_true')
    parser.add_argument('--framing-scale', type=float, default=1.0)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True, exist_ok=True)
    scale = args.framing_scale
    if scale <= 0:
        raise ValueError('Framing scale must be positive')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    groups = {}
    for name in ('base_frame', 'legs', 'armrests', 'seat_cushions', 'back_cushions'):
        before = set(bpy.data.objects)
        runpy.run_path(str(args.code / f'{name}.py'), run_name='__main__')
        bpy.context.view_layer.update()
        groups[name] = [obj for obj in bpy.data.objects if obj not in before and obj.type == 'MESH']
        if not groups[name]:
            raise ValueError(f'{name} did not create mesh objects')
    report = {name: measure(objects) for name, objects in groups.items()}
    (args.output / 'measurements.json').write_text(json.dumps(report, indent=2))
    if args.measure_only:
        return
    mesh_objects = [o for group in groups.values() for o in group]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.export_scene.gltf(filepath=str(args.output / 'sofa.glb'), export_format='GLB',
                              use_selection=True, export_yup=True, export_apply=True)
    # Object geometry remains in the generator's Z-up frame; export converts to Y-up.
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.15, 0.15, 0.15, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.5
    for name, position, energy in [('key', (1,-2,3), 180), ('fill', (-2,-1,2), 100), ('rim', (0,2,3), 140)]:
        data = bpy.data.lights.new(name, type='AREA')
        data.energy, data.shape, data.size = energy * scale**2, 'DISK', 2 * scale
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = Vector(position) * scale
        obj.rotation_euler = (Vector((0,0,.25))*scale-obj.location).to_track_quat('-Z','Y').to_euler()
    data = bpy.data.cameras.new('review_camera')
    camera = bpy.data.objects.new('review_camera', data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    data.type, data.ortho_scale = 'ORTHO', 1.24 * scale
    for name, position in [('front',(0,-3,.7)), ('three_quarter',(1.7,-2.5,1.3)), ('back',(1.7,2.5,1.2))]:
        camera.location = Vector(position) * scale
        camera.rotation_euler = (Vector((0,0,.25))*scale-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath = str(args.output / f'{name}.png')
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'sofa.blend'))


if __name__ == '__main__':
    main()
