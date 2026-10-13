"""Generate one saved-reference asset and fail if basic validation fails."""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import trimesh
from omegaconf import OmegaConf

from scenecode.agent_utils.code_object_generation import CodeObjectRunner
from scenecode.utils.openai import create_openai_client, encode_image_to_base64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    runner = CodeObjectRunner(OmegaConf.create({"asset_manager": {"code_object": {
        "config_path": str(args.config.resolve()), "max_attempts": 2, "max_concurrent_runs": 1}}}))
    started = time.monotonic()
    feedback = None
    for attempt in range(1, 4):
        result = runner.generate_from_image(image_path=args.image.resolve(),
                                           output_dir=args.output / f"attempt_{attempt}", feedback=feedback)
        scene = trimesh.load(result.mesh_path, force="scene")
        mesh = scene.to_geometry()
        extent = mesh.extents
        bounds = []
        for name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[name]
            node_mesh = scene.geometry[geometry_name].copy()
            node_mesh.apply_transform(transform)
            bounds.append((name, node_mesh.bounds))
        connected = {0} if bounds else set()
        gaps = []
        edges = []
        for i, (name_a, box_a) in enumerate(bounds):
            for j, (name_b, box_b) in enumerate(bounds[:i]):
                gap = float(np.linalg.norm(np.maximum(0, np.maximum(box_a[0] - box_b[1], box_b[0] - box_a[1]))))
                gaps.append({"a": name_a, "b": name_b, "aabb_gap_m": gap})
                if gap <= .002:
                    edges.append((i, j))
        for _ in bounds:
            for i, j in edges:
                if i in connected or j in connected:
                    connected.update([i, j])
        supports_connected = bool(bounds) and len(connected) == len(bounds)
        geometry_passed = (len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all()
                           and np.all(extent > .03) and np.all(extent < 5) and supports_connected)
        plan = json.loads(result.object_plan_path.read_text())
        if not result.full_object_render_path or not result.full_object_render_path.exists():
            raise RuntimeError("Missing final render")
        response = create_openai_client().chat.completions.create(
            model=args.model, response_format={"type": "json_object"}, messages=[{
                "role": "user", "content": [
                    {"type": "text", "text": "Compare reference (first image) and generated coffee table (second). "
                     "Check completeness, connected supports (legs, pedestal or plinth as in the reference) touching the tabletop and ground, reasonable "
                     "proportions, no floating/detached parts, and usable tabletop. Ignore decorative style, texture and camera-angle differences. "
                     "Do not infer scale from framing; use measured meter dimensions. Do not reject for exact visual-reference fidelity. "
                     f"Independent mesh inspection: dimensions (GLTF X,Y-up,Z)={extent.tolist()}, "
                     f"mesh components={[n for n, b in bounds]}, measured gaps={gaps}. "
                     "These bounds cannot prove surface quality; still inspect visible holes or missing tabletop surfaces. "
                     "Return JSON {\"acceptable\": boolean, \"reason\": string, \"suggestions\": string}."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode_image_to_base64(args.image)}},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode_image_to_base64(result.full_object_render_path)}}]}])
        critique = json.loads(response.choices[0].message.content)
        report = {"model": args.model, "attempt": attempt, "mesh": str(result.mesh_path), "render": str(result.full_object_render_path),
                  "extents_m": extent.tolist(), "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                  "plan_parts": len(plan["parts"]), "geometry_passed": bool(geometry_passed),
                  "part_aabb_gaps": gaps, "aabb_supports_connected": supports_connected,
                  "visual_critique": critique, "seconds": time.monotonic() - started,
                  "passed": bool(geometry_passed and critique.get("acceptable") is True),
                  "limits": "Bounds and visual checks do not establish room physics or articulation correctness."}
        (args.output / f"validation_{attempt}.json").write_text(json.dumps(report, indent=2))
        if report["passed"]:
            (args.output / "validation.json").write_text(json.dumps(report, indent=2))
            print("ASSET_PILOT_PASS", result.mesh_path, flush=True)
            return
        feedback = "Correct the previous candidate: " + json.dumps({"visual": critique,
            "measured_part_gaps_m": gaps, "supports_connected": supports_connected,
            "requirement": "All structural parts must touch. Use centered local mesh coordinates and apply PartPlan positions only once."})
    raise RuntimeError("All three asset candidates failed validation; room run blocked")


if __name__ == "__main__":
    main()
