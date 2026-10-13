"""Independent post-physics checks; a successful process exit alone is insufficient."""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.clearance_zones import compute_door_clearance_violations
from scenecode.agent_utils.physics_validation import compute_scene_collisions
from scenecode.agent_utils.reachability import compute_reachability
from scenecode.agent_utils.room import ObjectType
from scenecode.agent_utils.urdf_to_sdf import compute_sdf_bounding_box
from scenecode.furniture_agents.tools.scene_tools import SceneTools


def check_link_connectivity(path, tolerance=.025):
    """Conservative visual AABB connectivity at zero joint positions."""
    bounds = {link.get("name"): compute_sdf_bounding_box(path, link_name=link.get("name"))[:2]
              for link in ET.parse(path).findall("model/link") if link.findall("visual")}
    if not bounds:
        return {"passed": False, "disconnected_links": [], "tolerance_m": tolerance}
    reached = {next(iter(bounds))}
    while True:
        additions = set()
        for name, (lo, hi) in bounds.items():
            if name in reached:
                continue
            for neighbor in reached:
                other_lo, other_hi = bounds[neighbor]
                gap = np.maximum(np.maximum(lo - other_hi, other_lo - hi), 0)
                if np.linalg.norm(gap) <= tolerance:
                    additions.add(name)
                    break
        if not additions:
            break
        reached.update(additions)
    return {"passed": len(reached) == len(bounds),
            "disconnected_links": sorted(set(bounds) - reached), "tolerance_m": tolerance,
            "link_bounds": {k: [a.tolist(), b.tolist()] for k, (a, b) in bounds.items()}}


def check_joint_motion(room):
    """Sample each bounded 1-DOF joint while retaining the settled room pose."""
    from pydrake.all import DiagramBuilder, JointIndex
    from scenecode.agent_utils.drake_utils import create_drake_plant_and_scene_graph_from_scene
    builder = DiagramBuilder()
    plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=room, builder=builder, weld_furniture=False,
        free_mounted_objects_for_collision=True)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    graph_context = scene_graph.GetMyContextFromRoot(context)
    initial = plant.GetPositions(plant_context).copy()
    checks = []
    for index in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(index))
        if joint.num_positions() != 1:
            continue
        lower, upper = joint.position_lower_limits()[0], joint.position_upper_limits()[0]
        if not np.isfinite([lower, upper]).all():
            raise ValueError(f"Joint {joint.name()} has unbounded travel")
        samples = []
        for value in np.linspace(lower, upper, 5):
            positions = initial.copy()
            positions[joint.position_start()] = value
            plant.SetPositions(plant_context, positions)
            query = scene_graph.get_query_output_port().Eval(graph_context)
            pairs = query.ComputeSignedDistancePairwiseClosestPoints(0)
            maximum = max([-p.distance for p in pairs] + [0.])
            samples.append({"position": float(value), "max_penetration_m": float(maximum)})
        checks.append({"joint": joint.name(), "samples": samples,
                       "passed": all(s["max_penetration_m"] <= .01 for s in samples)})
    return checks


def validate(root):
    state_path = root / "scene_000/combined_house_after_furniture/house_state.json"
    report = {"passed": False, "state": str(state_path), "errors": [], "rooms": {}}
    try:
        for filename in ("house.blend", "house.dmd.yaml"):
            if not (state_path.parent / filename).is_file():
                report["errors"].append(f"Missing final export: {state_path.parent / filename}")
        state = json.loads(state_path.read_text())
        house = HouseScene.from_state_dict(state, house_dir=root / "scene_000")
        for room_id, room in house.rooms.items():
            objects = [obj for obj in room.objects.values() if obj.object_type == ObjectType.FURNITURE]
            if len(objects) != 4:
                report["errors"].append(f"{room_id}: expected exactly four furniture items; found {len(objects)}")
            matched = {}
            articulated_connectivity = {}
            patterns = {"sofa": r"sofa|couch", "coffee_table": r"coffee.?table|cocktail.?table",
                        "bookshelf": r"bookshelf|bookcase|book.?shelf", "tv_console": r"tv.?console|tv.?stand|media.?console|television.?console|entertainment.?center"}
            for category, pattern in patterns.items():
                matches = [obj for obj in objects if re.search(pattern, obj.name, re.I)]
                if len(matches) != 1:
                    report["errors"].append(f"{room_id}: expected one {category}; found {len(matches)}")
                else:
                    matched[category] = str(matches[0].object_id)
            for obj in objects:
                movable_joints = []
                extent = np.asarray(obj.bbox_max) - np.asarray(obj.bbox_min)
                if not np.isfinite(extent).all() or np.any(extent <= .01) or np.any(extent > 5):
                    report["errors"].append(f"{obj.name}: invalid bounds {extent.tolist()}")
                world_bounds = obj.compute_world_bounds()
                if world_bounds is None:
                    report["errors"].append(f"{obj.name}: missing world bounds")
                elif world_bounds[1][2] > room.room_geometry.wall_height + .01:
                    report["errors"].append(
                        f"{obj.name}: top {world_bounds[1][2]:.3f} m exceeds "
                        f"room height {room.room_geometry.wall_height:.3f} m")
                for path in (obj.geometry_path, obj.sdf_path):
                    if path is None or not Path(path).is_file():
                        report["errors"].append(f"{obj.name}: missing artifact {path}")
                if obj.sdf_path:
                    for joint in ET.parse(obj.sdf_path).findall(".//joint"):
                        if joint.get("type") in ("revolute", "prismatic"):
                            movable_joints.append(joint.get("name"))
                            lower = float(joint.findtext("axis/limit/lower", "nan"))
                            upper = float(joint.findtext("axis/limit/upper", "nan"))
                            if not np.isfinite([lower, upper]).all() or lower >= upper:
                                report["errors"].append(f"{obj.name}: invalid joint limits for {joint.get('name')}")
                if str(obj.object_id) == matched.get("tv_console") and not movable_joints:
                    report["errors"].append(f"{obj.name}: saved cabinet reference has movable doors, but no movable joint was exported")
                if movable_joints:
                    check = check_link_connectivity(obj.sdf_path)
                    articulated_connectivity[str(obj.object_id)] = check
                    if not check["passed"]:
                        report["errors"].append(f"{obj.name}: detached visual links at rest: {check['disconnected_links']}")
            collisions = compute_scene_collisions(room, penetration_threshold=.001, floor_penetration_tolerance=.01)
            door_violations = compute_door_clearance_violations(scene=room)
            if door_violations:
                report["errors"].append(f"{room_id}: {len(door_violations)} blocked door clearances")
            reachability = compute_reachability(room, robot_width=.5)
            motion = check_joint_motion(room)
            if any(not check["passed"] for check in motion):
                report["errors"].append(f"{room_id}: joint motion causes penetration exceeding 1 cm")
            facing = None
            if "sofa" in matched and "tv_console" in matched:
                facing = json.loads(SceneTools(room, OmegaConf.create({}))._check_facing_impl(
                    matched["sofa"], matched["tv_console"]))
                if not facing.get("is_facing"):
                    report["errors"].append(f"{room_id}: sofa does not face TV console")
            if collisions:
                report["errors"].append(f"{room_id}: {len(collisions)} collision violations")
            if not reachability.is_fully_reachable:
                report["errors"].append(f"{room_id}: disconnected walking space")
            report["rooms"][room_id] = {"matched_furniture": matched,
                "collisions": [c.to_description() for c in collisions],
                "door_clearance_violations": [c.to_description() for c in door_violations],
                "articulated_connectivity": articulated_connectivity,
                "reachability": json.loads(reachability.to_json()), "facing": facing, "joint_motion": motion}
        if not house.rooms:
            report["errors"].append("No rooms exported")
        report["passed"] = not report["errors"]
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["limitations"] = "Joint travel is sampled at five positions, not continuously; Drake collision filtering applies. Visual review remains necessary."
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = validate(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "independent_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
