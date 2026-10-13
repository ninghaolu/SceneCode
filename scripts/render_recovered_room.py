"""Render an exported room with CPU Cycles; cutaway affects preview only."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
args.output.mkdir(parents=True,exist_ok=True)
scene=bpy.context.scene
meshes=[o for o in scene.objects if o.type=='MESH']
points=[o.matrix_world@Vector(p) for o in meshes for p in o.bound_box]
lo=Vector([min(p[i] for p in points) for i in range(3)])
hi=Vector([max(p[i] for p in points) for i in range(3)])
center=(lo+hi)/2
target=Vector((center.x,center.y,lo.z+.6))
cutaway=[]
for obj in meshes:
    corners=[obj.matrix_world@Vector(p) for p in obj.bound_box]
    height=max(p.z for p in corners)-min(p.z for p in corners)
    if height>2.2:
        # Architecture only in this four-item room: tallest furniture is <2m.
        bm=bmesh.new()
        bm.from_mesh(obj.data)
        upper=[v for v in bm.verts if (obj.matrix_world@v.co).z>lo.z+.4]
        bmesh.ops.delete(bm,geom=upper,context='VERTS')
        bm.to_mesh(obj.data)
        bm.free()
        cutaway.append(obj.name)
scene.render.engine='CYCLES'
scene.cycles.device='CPU'
scene.cycles.samples=48
scene.cycles.use_denoising=True
scene.render.resolution_x=scene.render.resolution_y=768
scene.render.resolution_percentage=100
scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.25,.25,.25,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.7
for obj in list(scene.objects):
    if obj.type in ('CAMERA','LIGHT'):
        bpy.data.objects.remove(obj,do_unlink=True)
for name,delta,energy in [('key',(2,0,7),1700),('fill',(-3,3,6),1000)]:
    data=bpy.data.lights.new(name,'AREA')
    data.energy,data.size=energy,5
    obj=bpy.data.objects.new(name,data)
    scene.collection.objects.link(obj)
    obj.location=target+Vector(delta)
    obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
data=bpy.data.cameras.new('review_camera')
camera=bpy.data.objects.new('review_camera',data)
scene.collection.objects.link(camera)
scene.camera=camera
data.type,data.ortho_scale='ORTHO',max(hi.x-lo.x,hi.y-lo.y)*1.45
for name,delta in [('cutaway',(6,8,8)),('top',(0,0,10))]:
    camera.location=target+Vector(delta)
    camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(args.output/f'{name}.png')
    bpy.ops.render.render(write_still=True)
(args.output/'render_notes.json').write_text(json.dumps({'cutaway_architecture':cutaway,
    'note':'Upper architecture removed only in these previews. Exported house.blend and physics assets are unchanged.'},indent=2))
