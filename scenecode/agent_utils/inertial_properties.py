"""Finite rigid-body properties, with an explicit collision-volume approximation."""
import numpy as np
import trimesh


def compute_inertial_properties(visual_mesh, mass, collision_pieces):
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid mass: {mass}")
    mesh = visual_mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.fix_normals(multibody=True)

    def properties(candidate):
        volume = candidate.volume
        if not candidate.is_watertight or not np.isfinite(volume) or volume <= 1e-12:
            raise ValueError("Mesh is not a closed positive-volume solid")
        center = candidate.center_mass
        inertia = candidate.moment_inertia * (mass / volume)
        if not np.isfinite(center).all() or not np.isfinite(inertia).all():
            raise ValueError("Non-finite mass properties")
        if np.any(center < candidate.bounds[0] - 1e-6) or np.any(center > candidate.bounds[1] + 1e-6):
            raise ValueError("Center of mass lies outside geometry bounds")
        eigenvalues = np.linalg.eigvalsh(inertia)
        if np.any(eigenvalues <= 0) or eigenvalues[-1] > eigenvalues[:2].sum() + 1e-6:
            raise ValueError("Non-physical inertia tensor")
        return center, inertia

    reason = None
    basis = "visual_mesh"
    try:
        center, inertia = properties(mesh)
    except ValueError as exc:
        reason = str(exc)
        if not collision_pieces:
            raise ValueError("Invalid visual mass properties and no collision solids") from exc
        # VHACD pieces are the simulation's geometric approximation already.
        # Their convex hulls are closed, and each contributes positive mass.
        proxy = trimesh.util.concatenate([p.convex_hull for p in collision_pieces])
        center, inertia = properties(proxy)
        basis = "convex_collision_approximation"
    report = {"basis": basis, "visual_issue": reason, "mass_kg": mass,
              "center_of_mass": center.tolist(), "inertia_tensor": inertia.tolist(),
              "inertia_eigenvalues": np.linalg.eigvalsh(inertia).tolist(),
              "limitation": "Uniform density is assumed; collision-piece overlap may affect the approximation."}
    return center, inertia, report
