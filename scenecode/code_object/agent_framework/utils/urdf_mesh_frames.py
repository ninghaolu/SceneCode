"""Place Code_Object's world-coordinate OBJ exports in URDF link frames."""
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_exported_mesh_origins(text):
    """Enforce T_world_link * T_link_mesh = OBJ's Y-up to Z-up rotation.

    Only use for Code_Object OBJ exports, whose vertices already contain their
    assembled world positions. Joint frames and travel remain model-generated.
    """
    root = ET.fromstring(text)
    links = {link.get("name"): link for link in root.findall("link")}
    parents = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        if child in parents or child not in links or parent not in links:
            raise ValueError("Invalid URDF kinematic tree")
        origin = joint.find("origin")
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        parents[child] = (parent, transform)
    if len(set(links) - set(parents)) != 1:
        raise ValueError("URDF must have exactly one root link")

    def world(name, visiting=frozenset()):
        if name in visiting:
            raise ValueError("URDF kinematic cycle")
        if name not in parents:
            return np.eye(4)
        parent, transform = parents[name]
        return world(parent, visiting | {name}) @ transform

    obj_to_world = np.eye(4)
    obj_to_world[:3, :3] = Rotation.from_euler("x", np.pi / 2).as_matrix()
    for name, link in links.items():
        local = np.linalg.inv(world(name)) @ obj_to_world
        for element in [*link.findall("visual"), *link.findall("collision")]:
            mesh = element.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected exported OBJ mesh")
            path = PurePosixPath(mesh.get("filename", ""))
            if path.suffix.lower() != ".obj" or path.parts[:2] != ("blender_output", "parts") or ".." in path.parts:
                raise ValueError(f"Not a Code_Object world-coordinate export: {path}")
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if scale.shape != (3,) or not np.allclose(scale, 1):
                raise ValueError("Unexpected per-part scale on world-coordinate export")
            origin = element.find("origin")
            if origin is None:
                origin = ET.SubElement(element, "origin")
            origin.set("xyz", " ".join(f"{x:.12g}" for x in local[:3, 3]))
            origin.set("rpy", " ".join(f"{x:.12g}" for x in Rotation.from_matrix(local[:3, :3]).as_euler("xyz")))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
