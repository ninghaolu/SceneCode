"""Regression checks for the small-model pilot's failure modes."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import trimesh

from scenecode.agent_utils.urdf_to_sdf import compute_sdf_bounding_box
from scenecode.agent_utils.asset_router.router import AssetRouter
from scenecode.agent_utils.saved_images import SavedImageGenerator
from scenecode.utils.model_audit import _request, _response


def test_sdf_bounds_follow_uris_node_visual_and_link_transforms_without_double_scaling(tmp_path):
    box = trimesh.creation.box(extents=[1, 2, 3])
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [2, 0, 0]
    scene.add_geometry(box, transform=transform)
    (tmp_path / "box.glb").write_bytes(scene.export(file_type="glb"))
    sdf = tmp_path / "model.sdf"
    sdf.write_text('''<sdf version="1.9"><model name="m">
      <link name="base"><pose>3 0 0 0 0 0</pose></link>
      <link name="body"><pose relative_to="base">1 0 0 0 0 0</pose>
      <visual name="v"><pose>0 0 2 0 0 0</pose><geometry><mesh>
      <uri>box.glb</uri><scale>2 2 2</scale></mesh></geometry></visual></link>
      </model></sdf>''')
    minimum, maximum, center = compute_sdf_bounding_box(sdf, scale_factor=2)
    np.testing.assert_allclose(maximum - minimum, [2, 6, 4])
    np.testing.assert_allclose(center, [8, 0, 2])


def test_empty_sdf_is_error_not_zero_bounds(tmp_path):
    sdf = tmp_path / "empty.sdf"
    sdf.write_text('<sdf><model name="m"><link name="base"/></model></sdf>')
    with pytest.raises(ValueError, match="No visual mesh"):
        compute_sdf_bounding_box(sdf)


def test_articulated_connectivity_rejects_detached_door(tmp_path):
    from scripts.validate_qwen_room import check_link_connectivity
    trimesh.creation.box(extents=[1, 1, 1]).export(tmp_path / "body.obj")
    sdf = tmp_path / "test.sdf"
    template = '<sdf><model name="m"><link name="body"><visual><geometry><mesh><uri>body.obj</uri></mesh></geometry></visual></link><link name="door"><pose relative_to="body">{x} 0 0 0 0 0</pose><visual><geometry><mesh><uri>body.obj</uri></mesh></geometry></visual></link></model></sdf>'
    sdf.write_text(template.format(x=1.01))
    assert check_link_connectivity(sdf)["passed"]
    sdf.write_text(template.format(x=1.2))
    assert check_link_connectivity(sdf)["disconnected_links"] == ["door"]


def test_world_obj_origin_correction_preserves_nested_joint_assembly():
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from scenecode.code_object.agent_framework.utils.urdf_mesh_frames import normalize_exported_mesh_origins
    xml = '<robot><link name="base"/><link name="body"/><link name="door"><visual><origin xyz="4 5 6"/><geometry><mesh filename="blender_output/parts/door.obj"/></geometry></visual></link><joint name="j1" type="fixed"><parent link="base"/><child link="body"/><origin xyz="1 2 3" rpy="0 0 .4"/></joint><joint name="j2" type="prismatic"><parent link="body"/><child link="door"/><origin xyz=".3 .2 .1" rpy=".1 .2 .3"/></joint></robot>'
    root = ET.fromstring(normalize_exported_mesh_origins(xml))
    def transform(origin):
        t = np.eye(4)
        t[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        return t
    assembled = transform(root.find('joint[@name="j1"]/origin')) @ transform(root.find('joint[@name="j2"]/origin')) @ transform(root.find('link[@name="door"]/visual/origin'))
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("x", np.pi / 2).as_matrix()
    np.testing.assert_allclose(assembled, expected, atol=1e-10)


@pytest.mark.parametrize("kind, expected", [("prismatic", .1), ("revolute", .2)])
def test_joint_travel_scales_only_linear_coordinates(kind, expected):
    import xml.etree.ElementTree as ET
    from scenecode.agent_utils.urdf_to_sdf import convert_urdf_joint_to_sdf
    joint = ET.fromstring(f'<joint name="j" type="{kind}"><parent link="b"/><child link="c"/><limit lower="-.2" upper=".2" velocity=".2" effort="5"/></joint>')
    output = convert_urdf_joint_to_sdf(joint, scale_factor=.5)
    assert float(output.findtext('axis/limit/upper')) == expected
    assert float(output.findtext('axis/limit/lower')) == -expected
    assert float(output.findtext('axis/limit/velocity')) == expected


def test_conversion_does_not_silently_accept_failed_mesh_merge(tmp_path):
    import xml.etree.ElementTree as ET
    from scenecode.agent_utils.urdf_to_sdf import merge_link_visual_meshes_for_sdf
    (tmp_path / "part.obj").write_text("dummy")
    link = ET.fromstring('<link name="body"><visual><geometry><mesh filename="part.obj"/></geometry></visual></link>')
    with patch("scenecode.agent_utils.urdf_to_sdf.merge_objs_to_gltf", side_effect=RuntimeError("worker failed")):
        with pytest.raises(RuntimeError, match="Failed to merge visual meshes"):
            merge_link_visual_meshes_for_sdf(link, tmp_path, tmp_path, "body")


def test_pedestal_table_is_complete_but_unsupported_top_is_not():
    from scenecode.code_object.agent_framework.agents.planner_checker import PlannerCheckerAgent
    checker = object.__new__(PlannerCheckerAgent)
    checker.part_templates = {"table": {"required": ["tabletop", "legs"]}}
    plan = SimpleNamespace(category="table", parts=[SimpleNamespace(name="top", part_type="tabletop"),
                                                    SimpleNamespace(name="plinth", part_type="base")])
    assert not checker._check_parts_completeness(plan)
    plan.parts.pop()
    assert checker._check_parts_completeness(plan)[0].severity == "error"


@pytest.mark.asyncio
async def test_checker_receives_computed_bounds_and_failure_is_not_a_pass():
    from scenecode.code_object.agent_framework.agents.planner_checker import PlannerCheckerAgent
    checker = object.__new__(PlannerCheckerAgent)
    checker.llm_client = object()
    checker.prompt_template = "Check geometry"
    checker.logger = MagicMock()
    checker._call_llm = AsyncMock(return_value='{"issues": [], "corrections": {}}')
    part = {"name": "base", "part_type": "base", "position": {"x": 0., "y": 0., "z": .075},
            "shape": {"dimensions": {"width": .9, "depth": .5, "height": .15}}}
    plan = SimpleNamespace(name="table", parts=[SimpleNamespace(to_dict=lambda: part)], to_json=lambda: "{}")
    await checker._llm_based_check(plan, text_input="table")
    delivered = checker._call_llm.call_args.args[0][-1]["content"][-1]["text"]
    assert '"bounds_min": {"x": -0.45, "y": -0.25, "z": 0.0}' in delivered
    checker._rule_based_check = MagicMock(return_value=([], {}))
    checker._llm_based_check = AsyncMock(side_effect=RuntimeError("API unavailable"))
    result = await checker.run(plan, text_input="table")
    assert not result.is_valid
    assert result.issues[0].category == "validation_failure"


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_stages", [[], ["plan_check"]])
async def test_strict_plan_gate_blocks_construction_even_when_resuming(tmp_path, skip_stages):
    from scenecode.code_object.agent_framework.pipeline import Pipeline
    pipeline = object.__new__(Pipeline)
    pipeline.config = {"pipeline": {"enable_planner_check": True, "require_valid_plan": True}}
    pipeline.logger = MagicMock()
    pipeline._connect_blender = AsyncMock(return_value=True)
    pipeline._clear_blender_scene = AsyncMock()
    pipeline._disconnect_blender = AsyncMock()
    plan = SimpleNamespace(name="table", save=MagicMock(), to_dict=lambda: {"name": "table"})
    pipeline._check_existing_plan = MagicMock(return_value=plan)
    pipeline.planner = MagicMock()
    pipeline.planner_checker = SimpleNamespace(check_and_fix=AsyncMock(return_value=(plan, [SimpleNamespace(is_valid=False)])))
    pipeline.part_constructor = SimpleNamespace(run=AsyncMock())
    result = await pipeline.run(text_input="table", output_dir=str(tmp_path), skip_stages=skip_stages)
    assert not result.success
    assert "Plan still has checker errors" in result.error
    pipeline.planner_checker.check_and_fix.assert_awaited_once()
    pipeline.part_constructor.run.assert_not_awaited()


def test_rejected_final_candidate_is_not_accepted_and_feedback_is_forwarded(tmp_path):
    router = object.__new__(AssetRouter)
    router._generate_code_object_geometry = MagicMock(return_value=SimpleNamespace(geometry_path=tmp_path / "mesh.glb", image_path=tmp_path / "reference.png"))
    router.validate_asset = MagicMock(return_value=SimpleNamespace(is_acceptable=False, reason="Detached legs", suggestions="Connect legs to top"))
    item = SimpleNamespace(description="coffee table", short_name="coffee_table")
    result = router._try_code_generated_strategy(item, 2, MagicMock(), MagicMock(), tmp_path, tmp_path, tmp_path)
    assert result is None
    assert router.validate_asset.call_count == 2
    assert router.validate_asset.call_args.kwargs['reference_image'] == tmp_path / 'reference.png'
    assert "Detached legs" in router._generate_code_object_geometry.call_args.kwargs["feedback"]
    assert len(list(tmp_path.glob("*/feedback_for_next_attempt.txt"))) == 2


def test_articulated_candidate_also_requires_validation_on_last_attempt(tmp_path):
    from scenecode.agent_utils.asset_router.dataclasses import CodeArticulatedGeometry
    mesh = tmp_path / "candidate.glb"
    mesh.write_bytes(b"test")
    item = SimpleNamespace(description="TV console", short_name="tv_console")
    router = object.__new__(AssetRouter)
    router._generate_code_articulated_geometry = MagicMock(return_value=CodeArticulatedGeometry(
        urdf_path=tmp_path / "candidate.urdf", item=item, geometry_path=mesh))
    router.validate_asset = MagicMock(return_value=SimpleNamespace(is_acceptable=False,
        reason="Missing drawer", suggestions="Add independent drawer"))
    assert router._try_code_articulated_strategy(item, 2, MagicMock(), MagicMock(), tmp_path, tmp_path, tmp_path) is None
    assert router.validate_asset.call_count == 2
    assert "Missing drawer" in router._generate_code_articulated_geometry.call_args.kwargs["feedback"]


def test_audit_rejects_model_fallback_and_supplies_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENECODE_REQUIRED_MODEL", "qwen3.5-flash")
    monkeypatch.setenv("SCENECODE_AUDIT_DIR", str(tmp_path))
    request = httpx.Request("POST", "https://example.test/responses", json={
        "model": "qwen3.5-flash", "input": "classify",
        "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}}})
    _request(request)
    assert "JSON Schema" in json.loads(request.content)["instructions"]
    assert json.loads(request.content)["reasoning"]["effort"] == "none"
    response = httpx.Response(200, json={"model": "other-model"}, request=request)
    with pytest.raises(RuntimeError, match="Unexpected returned model"):
        _response(response)
    assert "api_seconds" in json.loads(next(tmp_path.glob("*.json")).read_text())
    with pytest.raises(RuntimeError, match="Unintended model"):
        _request(httpx.Request("POST", "https://example.test/responses", json={"model": "gpt-5"}))


def test_saved_reference_is_exact_and_unknown_asset_fails(tmp_path):
    import hashlib
    source = tmp_path / "source.png"
    source.write_bytes(b"unchanged reference")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"aliases": ["sofa"], "path": str(source),
                                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]))
    generator = SavedImageGenerator(manifest)
    output = tmp_path / "sofa_123.png"
    generator.generate_images("modern", ["a sofa"], [output])
    assert output.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="matched 0"):
        generator.generate_images("modern", ["a lamp"], [tmp_path / "lamp.png"])


def test_qwen_json_fence_normalization_preserves_raw_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENECODE_REQUIRED_MODEL", "qwen3.5-plus")
    monkeypatch.setenv("SCENECODE_AUDIT_DIR", str(tmp_path))
    request = httpx.Request("POST", "https://example.test/responses", json={
        "model": "qwen3.5-plus", "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}}})
    _request(request)
    raw = '```json\n{"answer": 42}\n```'
    response = httpx.Response(200, request=request, json={"model": "qwen3.5-plus", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": raw}]}]})
    _response(response)
    assert json.loads(response.json()["output"][0]["content"][0]["text"]) == {"answer": 42}
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["response"]["output"][0]["content"][0]["text"] == raw
    assert record["normalization"] == "removed_markdown_json_fence"


@pytest.mark.asyncio
async def test_cached_part_is_geometry_checked_and_repaired(tmp_path):
    bpy = pytest.importorskip("bpy")
    from scenecode.code_object.agent_framework.agents.part_constructor import PartConstructorAgent
    constructor = object.__new__(PartConstructorAgent)
    constructor.config = {"pipeline": {"validate_part_positions": True}}
    constructor.logger = MagicMock()
    constructor.max_retry = 2
    constructor.save_action_trace = False
    constructor.llm_client = object()
    constructor._validate_generated_code_for_part = MagicMock(return_value=None)
    constructor._generate_code_with_llm = AsyncMock()
    prefix = "import bpy\nif bpy.data.objects.get('guard_test'):\n    bpy.data.objects.remove(bpy.data.objects['guard_test'], do_unlink=True)\n"
    bad = prefix + "bpy.ops.mesh.primitive_cube_add(size=1, location=(.2,0,0))\nbpy.context.object.name='guard_test'\n"
    good = bad.replace("(.2,0,0)", "(0,0,0)")
    (tmp_path / "guard_test.py").write_text(bad)
    constructor._generate_code_fix_from_error = AsyncMock(return_value=good)

    async def execute(code):
        try:
            exec(code, {})
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    constructor._execute_code = execute
    part = SimpleNamespace(name="guard_test", instances=[], to_dict=lambda: {
        "name": "guard_test", "position": {"x": 0, "y": 0, "z": 0},
        "shape": {"dimensions": {"width": 1, "depth": 1, "height": 1}}})
    result = await constructor.construct_part(part, image_path=None, text_input="cube", output_dir=str(tmp_path))
    assert result["execution_result"]["success"]
    assert result["action_trace"].iterations == 2
    constructor._generate_code_with_llm.assert_not_awaited()
    assert "Geometry placement mismatch" in constructor._generate_code_fix_from_error.call_args.args[2]
    assert (tmp_path / "guard_test.py").read_text() == good
    bpy.data.objects.remove(bpy.data.objects["guard_test"], do_unlink=True)


def test_geometry_guard_accepts_declared_mirror_but_rejects_vertical_shift():
    bpy = pytest.importorskip("bpy")
    from mathutils import Matrix
    from scenecode.code_object.agent_framework.utils.geometry_guard import part_position_guard
    bpy.ops.mesh.primitive_cube_add(size=.1)
    obj = bpy.context.object
    obj.name = "mirror_guard_test"
    obj.data.transform(Matrix.Translation((-.4, 0, 0)))
    obj.modifiers.new(name="mirror", type="MIRROR")
    part = SimpleNamespace(to_dict=lambda: {"name": "mirror_guard_test", "position": {"x": -.4, "y": 0, "z": 0},
        "is_symmetric": True, "symmetric_axis": "x", "shape": {"dimensions": {"width": .1, "depth": .1, "height": .1}}})
    guard = part_position_guard(part)
    exec(guard, {})
    obj.location.z = .2
    with pytest.raises(ValueError, match="Geometry placement mismatch"):
        exec(guard, {})
    bpy.data.objects.remove(obj, do_unlink=True)


def test_geometry_guard_checks_array_template_component():
    bpy = pytest.importorskip("bpy")
    from scenecode.code_object.agent_framework.utils.geometry_guard import part_position_guard
    bpy.ops.mesh.primitive_cube_add(size=.1, location=(-.2, -.2, 0))
    obj = bpy.context.object
    obj.name = "array_guard_test"
    for axis in (0, 1):
        mod = obj.modifiers.new(name=f"array_{axis}", type="ARRAY")
        mod.count = 2
        mod.relative_offset_displace = (4, 0, 0) if axis == 0 else (0, 4, 0)
    part = SimpleNamespace(to_dict=lambda: {"name": "array_guard_test", "position": {"x": -.2, "y": -.2, "z": 0},
        "shape": {"dimensions": {"width": .1, "depth": .1, "height": .1}, "modifiers": ["array_x_2", "array_y_2"]}})
    guard = part_position_guard(part)
    exec(guard, {})
    obj.location.z = .2
    with pytest.raises(ValueError, match="Geometry placement mismatch"):
        exec(guard, {})
    bpy.data.objects.remove(obj, do_unlink=True)


def test_inertia_repairs_winding_and_uses_explicit_closed_collision_fallback():
    from scenecode.agent_utils.inertial_properties import compute_inertial_properties
    box = trimesh.creation.box(extents=[1, 2, 3])
    box.invert()
    center, inertia, report = compute_inertial_properties(box, 12, [])
    np.testing.assert_allclose(center, [0, 0, 0])
    np.testing.assert_allclose(np.diag(inertia), [13, 10, 5])
    assert report["basis"] == "visual_mesh"
    broken = trimesh.creation.box()
    broken.update_faces(np.arange(len(broken.faces) - 2))
    with pytest.raises(ValueError, match="no collision solids"):
        compute_inertial_properties(broken, 12, [])
    _, inertia, report = compute_inertial_properties(broken, 12, [trimesh.creation.box()])
    assert report["basis"] == "convex_collision_approximation"
    assert np.all(np.linalg.eigvalsh(inertia) > 0)


def test_uniform_gltf_scaling_compares_height_to_y_up_axis(tmp_path):
    from scenecode.agent_utils.mesh_utils import scale_mesh_uniformly_to_dimensions
    source = tmp_path / "shelf.glb"
    source.write_bytes(trimesh.Scene(trimesh.creation.box(extents=[.35, 1, .4])).export(file_type="glb"))
    output, scale = scale_mesh_uniformly_to_dimensions(source, [1, .35, 2], output_path=tmp_path / "scaled.glb")
    assert scale == pytest.approx(2)
    np.testing.assert_allclose(trimesh.load(output, force="scene").to_geometry().extents, [.7, 2, .8], atol=1e-6)


def test_sdf_massless_anchor_keeps_zero_inertia(tmp_path):
    import xml.etree.ElementTree as ET
    from scenecode.utils.inertia_utils import fix_sdf_file_inertia
    path = tmp_path / "anchor.sdf"
    path.write_text('<sdf><model><link><inertial><mass>0</mass><inertia>'
                    '<ixx>1e-10</ixx><iyy>1e-10</iyy><izz>1e-10</izz>'
                    '<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>'
                    '</inertia></inertial></link></model></sdf>')
    assert fix_sdf_file_inertia(path)
    assert all(float(element.text) == 0 for element in ET.parse(path).find('.//inertia'))
    assert not fix_sdf_file_inertia(path)


def test_real_blender_guard_detects_join_origin_offset():
    bpy = pytest.importorskip("bpy")
    from scenecode.code_object.agent_framework.utils.geometry_guard import part_position_guard
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    panels = []
    for z in (-.19, .19):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        panel = bpy.context.object
        panel.scale = (1, .35, .02)
        bpy.ops.object.transform_apply(scale=True)
        panel.location.z = z
        panels.append(panel)
    bpy.ops.object.select_all(action="DESELECT")
    for panel in panels:
        panel.select_set(True)
    bpy.context.view_layer.objects.active = panels[-1]
    bpy.ops.object.join()
    body = bpy.context.object
    body.name = "cabinet_body"
    body.location.z = .24
    part = SimpleNamespace(to_dict=lambda: {"name": "cabinet_body", "position": {"x": 0, "y": 0, "z": .24},
        "shape": {"dimensions": {"width": 1, "depth": .35, "height": .4}}})
    guard = part_position_guard(part)
    with pytest.raises(ValueError, match="Geometry placement mismatch"):
        exec(guard, {})
    # Correct the retained top-panel origin, then set the intended world center.
    bpy.context.scene.cursor.location = (0, 0, .05)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    body.location = (0, 0, .24)
    exec(guard, {})
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
