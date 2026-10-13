"""Repair legacy rigid SDF mass properties, preserving the original SDF as evidence."""
import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from scenecode.agent_utils.inertial_properties import compute_inertial_properties


def repair(path):
    tree = ET.parse(path)
    model = tree.getroot().find("model")
    links = model.findall("link")
    if len(links) != 1 or model.findall("joint"):
        raise ValueError("This repair supports only generated single-link rigid SDFs")
    link = links[0]
    for element in [link, *link.findall("visual"), *link.findall("collision")]:
        if element.find("pose") is not None:
            raise ValueError("Unexpected link/geometry pose; cannot assume the mesh frame")
    if any(e.text.strip() != "1 1 1" for e in link.findall(".//mesh/scale")):
        raise ValueError("Unexpected mesh scaling")
    uri = link.findtext("visual/geometry/mesh/uri")
    visual = trimesh.load(path.parent / uri, force="scene").to_geometry()
    visual.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    pieces = [trimesh.load(path.parent / e.text, force="mesh") for e in link.findall("collision/geometry/mesh/uri")]
    mass = float(link.findtext("inertial/mass"))
    center, tensor, report = compute_inertial_properties(visual, mass, pieces)
    backup = path.with_name(path.name + f".before_inertia_repair_{time.time_ns()}")
    backup.write_bytes(path.read_bytes())
    inertial = link.find("inertial")
    pose = inertial.find("pose")
    if pose is None:
        pose = ET.SubElement(inertial, "pose")
    pose.text = " ".join(str(float(v)) for v in center) + " 0 0 0"
    old = inertial.find("inertia")
    if old is not None:
        inertial.remove(old)
    inertia = ET.SubElement(inertial, "inertia")
    for name, i, j in [("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2), ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2)]:
        ET.SubElement(inertia, name).text = str(float(tensor[i, j]))
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    report["original_sdf"] = str(backup)
    path.with_suffix(".inertia.json").write_text(json.dumps(report, indent=2))
    print(path, report["basis"], "COM", center.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    for path in parser.parse_args().paths:
        repair(path)
