"""Recover a saved room checkpoint without new generation or quality claims."""
import argparse
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf

from scenecode.agent_utils.house import HouseLayout, HouseScene, RoomGeometry
from scenecode.agent_utils.physical_feasibility import apply_physical_feasibility_postprocessing
from scenecode.agent_utils.room import ObjectType, RoomScene
from scripts.validate_qwen_room import validate


def recover(source: Path, output: Path, checkpoint: str, defer_blend: bool = False):
    source, output = source.resolve(), output.resolve()
    room_source = source / "room/scene_000/room_living_room"
    checkpoint_path = room_source / checkpoint
    if (output / "room").exists():
        raise FileExistsError(f"Refusing to overwrite recovery: {output}")
    state = json.loads(checkpoint_path.read_text())
    cfg = OmegaConf.to_container(OmegaConf.load(source / "room/.hydra/config.yaml"), resolve=True)
    output.mkdir(parents=True, exist_ok=True)
    record = {"kind": "checkpoint_recovery_not_fresh_baseline", "source": str(source),
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "started_at": time.time(), "state": "running", "new_model_calls": 0,
              "interventions": ["Resume saved layout and geometry; skip interrupted LLM critique; run original physics and exports"],
              "visual_review": "pending; saved sofa has known reference mismatch"}
    record_path = output / "recovery.json"
    record_path.write_text(json.dumps(record, indent=2))
    try:
        house_dir = output / "room/scene_000"
        room_dir = house_dir / "room_living_room"
        room_dir.mkdir(parents=True)
        for name in ("room_geometry",):
            shutil.copytree(source / "room/scene_000" / name, house_dir / name)
        shutil.copytree(room_source / "generated_assets", room_dir / "generated_assets")
        shutil.copy2(source / "room/scene_000/house_layout.json", house_dir / "house_layout.json")
        shutil.copy2(checkpoint_path, room_dir / "checkpoint_before_physics.json")
        OmegaConf.save(OmegaConf.create(cfg), output / "recovery_config.yaml")
        room = RoomScene(room_geometry=RoomGeometry.from_dict(state["room_geometry"], scene_dir=room_dir),
                         scene_dir=room_dir, room_id="living_room",
                         text_description=state.get("text_description", ""))
        room.restore_from_state_dict(state)
        projection = cfg["experiment"]["projection"]
        furniture, simulation = projection["furniture"], projection["simulation"]
        physics = cfg["furniture_agent"]["physics_validation"]
        room, success, removed = apply_physical_feasibility_postprocessing(
            scene=room, weld_furniture=False,
            projection_enabled=projection["enabled"] and furniture["enabled"],
            projection_influence_distance=furniture["influence_distance"],
            projection_solver_name=furniture["solver_name"],
            projection_iteration_limit=furniture["iteration_limit"],
            projection_time_limit_s=furniture["time_limit_s"],
            projection_xy_only=furniture["xy_only"], projection_fix_rotation=furniture["fix_rotation"],
            simulation_enabled=simulation["enabled"], simulation_time_s=simulation["simulation_time_s"],
            simulation_time_step_s=simulation["time_step_s"], simulation_timeout_s=simulation["timeout_s"],
            simulation_html_path=None, remove_fallen_furniture=physics["remove_fallen_furniture"],
            fallen_tilt_threshold_degrees=physics["fallen_tilt_threshold_degrees"])
        record.update(physics_pipeline_returned_success=bool(success), removed_objects=[str(x) for x in removed])
        (room_dir / "scene_after_physics.json").write_text(json.dumps(room.to_state_dict(), indent=2))
        layout = HouseLayout.from_dict(json.loads((house_dir / "house_layout.json").read_text()), house_dir=house_dir)
        HouseScene(layout=layout, rooms={"living_room": room}).assemble(
            cfg=None if defer_blend else cfg, output_name="combined_house_after_furniture", include_object_types=[ObjectType.FURNITURE])
        if defer_blend:
            record.update(state="ready_for_gpu_export", numeric_validation_passed=None)
            return True
        report = validate(output / "room")
        (output / "room/independent_validation.json").write_text(json.dumps(report, indent=2))
        record.update(state="exported_pending_visual_review" if report["passed"] else "validation_failed",
                      numeric_validation_passed=report["passed"], errors=report["errors"])
        return report["passed"]
    except Exception as exc:
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_at"] = time.time()
        record_path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default="scene_renders/furniture/renders_006/scene_state.json")
    parser.add_argument("--defer-blend", action="store_true", help="Save settled state; export and validate in a separate GPU job")
    args = parser.parse_args()
    raise SystemExit(0 if recover(args.source, args.output, args.checkpoint, args.defer_blend) else 1)
