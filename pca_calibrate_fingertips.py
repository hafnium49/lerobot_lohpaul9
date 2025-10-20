#!/usr/bin/env python3
import argparse
import numpy as np
import trimesh
from lxml import etree as ET
import mujoco  # only used when --verify is passed

# ---------- math utils ----------
def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)

def mat_to_quat(R):
    # robust conversion (MuJoCo uses w,x,y,z)
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z])
    # normalize
    q /= np.linalg.norm(q) + 1e-12
    return q

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

# ---------- PCA fingertip frame from STL ----------
def pca_frame_from_stl(stl_path, axis="z", sign=+1, thresh_deg=25, min_pts=100):
    """
    Returns (c_mesh, q_mesh, n_mesh)
    - c_mesh: 3D center of the fingertip contact patch (mesh coords)
    - q_mesh: quaternion (w,x,y,z) whose columns form [t1, t2, n] (mesh coords)
    - n_mesh: unit normal (mesh coords)
    Patch selection: vertices whose normals are within 'thresh_deg' of +axis * sign
    """
    mesh = trimesh.load_mesh(stl_path, process=True)
    v = mesh.vertices
    n = mesh.vertex_normals

    axis_vec = {"x":np.array([1,0,0]),
                "y":np.array([0,1,0]),
                "z":np.array([0,0,1])}[axis.lower()] * float(sign)
    cos_th = np.cos(np.deg2rad(thresh_deg))
    sel = (n @ axis_vec) > cos_th
    P = v[sel]
    if P.shape[0] < min_pts:
        # fallback: use all vertices
        P = v

    c = P.mean(axis=0)                    # centroid
    X = P - c
    C = (X.T @ X) / max(1, len(P))        # covariance
    eigvals, U = np.linalg.eigh(C)        # ascending order
    U = U[:, ::-1]                        # columns: t1, t2, t3 (max→min)
    t1, t2 = U[:,0], U[:,1]
    nrm = np.cross(t1, t2)                # right-handed normal
    nrm /= np.linalg.norm(nrm) + 1e-12

    # stabilize outward normal: prefer +axis_vec direction (flip if needed)
    if (nrm @ axis_vec) < 0:
        nrm = -nrm
        t2 = -t2  # keep right-handed

    # orthonormalize
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(nrm, t1); t2 /= np.linalg.norm(t2) + 1e-12
    R = np.stack([t1, t2, nrm], axis=1)
    q = mat_to_quat(R)
    return c.astype(float), q.astype(float), nrm.astype(float)

# ---------- XML helpers ----------
def parse_float_list(s, n):
    if s is None:
        return np.zeros(n, dtype=float)
    x = np.fromstring(s, sep=' ', dtype=float)
    if x.size != n:
        # allow comma-separated or extra whitespace
        x = np.array([float(tok) for tok in s.replace(',', ' ').split()])
    return x

def find_geom_by_mesh(xml_root, mesh_name):
    """
    Returns (geom_elem, parent_body_elem)
    """
    for body in xml_root.iterfind(".//body"):
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh" and geom.get("mesh") == mesh_name:
                return geom, body
    raise ValueError(f"Mesh geom with mesh=\"{mesh_name}\" not found.")

# ---------- main calibrator ----------
def main():
    ap = argparse.ArgumentParser(description="PCA-based fingertip calibrator for MuJoCo (SO-101)")
    ap.add_argument("--xml", required=True, help="Path to MuJoCo XML")
    ap.add_argument("--fixed-stl", required=True, help="STL for fixed jaw (wrist follower)")
    ap.add_argument("--moving-stl", required=True, help="STL for moving jaw")
    ap.add_argument("--fixed-mesh", required=True, help="mesh name used in <geom mesh=\"...\"> for fixed jaw")
    ap.add_argument("--moving-mesh", required=True, help="mesh name used in <geom mesh=\"...\"> for moving jaw")
    ap.add_argument("--radius", type=float, default=0.004, help="sphere radius (m)")
    ap.add_argument("--fixed-axis", default="z", choices=["x","y","z"], help="dominant normal axis in the STL of fixed tip")
    ap.add_argument("--moving-axis", default="z", choices=["x","y","z"], help="dominant normal axis in the STL of moving tip")
    ap.add_argument("--fixed-sign", type=int, default=+1, choices=[-1, +1], help="+1 if fingertip normal roughly +axis; -1 if roughly -axis")
    ap.add_argument("--moving-sign", type=int, default=-1, choices=[-1, +1], help="+1 if fingertip normal roughly +axis; -1 if roughly -axis")
    ap.add_argument("--thresh-deg", type=float, default=25.0, help="normal threshold (deg) for patch extraction")
    ap.add_argument("--closed-angle", type=float, default=0.0, help="gripper joint angle (rad) considered 'closed' for verification")
    ap.add_argument("--verify", action="store_true", help="load XML and report world-space center distance")
    args = ap.parse_args()

    # 1) PCA on both STLs (mesh coords)
    c_fix_m, q_fix_m, n_fix_m = pca_frame_from_stl(args.fixed_stl, axis=args.fixed_axis, sign=args.fixed_sign, thresh_deg=args.thresh_deg)
    c_mov_m, q_mov_m, n_mov_m = pca_frame_from_stl(args.moving_stl, axis=args.moving_axis, sign=args.moving_sign, thresh_deg=args.thresh_deg)

    # 2) Parse XML, get mesh->body transforms
    tree = ET.parse(args.xml)
    root = tree.getroot()

    g_fix, b_fix = find_geom_by_mesh(root, args.fixed_mesh)
    g_mov, b_mov = find_geom_by_mesh(root, args.moving_mesh)

    # mesh geom pose in BODY frame
    p_fix_b = parse_float_list(g_fix.get("pos"), 3)
    q_fix_b = parse_float_list(g_fix.get("quat"), 4) if g_fix.get("quat") is not None else np.array([1,0,0,0], float)
    p_mov_b = parse_float_list(g_mov.get("pos"), 3)
    q_mov_b = parse_float_list(g_mov.get("quat"), 4) if g_mov.get("quat") is not None else np.array([1,0,0,0], float)
    R_fix_b = quat_to_mat(q_fix_b)
    R_mov_b = quat_to_mat(q_mov_b)

    # 3) Transform PCA result from mesh->body frame
    # site pose (pos/quat) for fixed tip, expressed in BODY (gripper) frame
    p_site_fix_b = p_fix_b + R_fix_b @ c_fix_m
    R_site_fix_b = R_fix_b @ quat_to_mat(q_fix_m)
    q_site_fix_b = mat_to_quat(R_site_fix_b)
    n_fix_b = R_fix_b @ n_fix_m

    # site pose for moving tip, expressed in BODY (moving jaw) frame
    p_site_mov_b = p_mov_b + R_mov_b @ c_mov_m
    R_site_mov_b = R_mov_b @ quat_to_mat(q_mov_m)
    q_site_mov_b = mat_to_quat(R_site_mov_b)
    n_mov_b = R_mov_b @ n_mov_m

    # 4) Sphere centers in BODY frames: offset by radius along normals so they face each other
    # By convention: push fixed sphere "outward" (+n_fix_b), moving sphere "outward" (+n_mov_b).
    # If those normals don't face each other in the assembled robot, the verify pass will tell you (and you can flip signs).
    r = float(args.radius)
    p_sphere_fix_b = p_site_fix_b + r * n_fix_b
    p_sphere_mov_b = p_site_mov_b + r * n_mov_b

    # 5) Print XML snippets to paste into the corresponding BODY blocks
    def fmt(v): return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
    def fmtq(q): return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"

    print("\n=== Fixed jaw (paste INSIDE the body that contains the fixed-jaw mesh geom) ===")
    print(f'<site name="tip_fixed_pca" pos="{fmt(p_site_fix_b)}" quat="{fmtq(q_site_fix_b)}" size="0.003" rgba="0 1 0 0.6"/>')
    print(f'<geom name="fixed_fingertip" type="sphere" size="{r:.6f}" pos="{fmt(p_sphere_fix_b)}" '
          f'class="task_objects" material="gripper_rubber" friction="1.0 0.003 0.0001" contype="0" conaffinity="2"/>')

    print("\n=== Moving jaw (paste INSIDE the body that contains the moving-jaw mesh geom) ===")
    print(f'<site name="tip_moving_pca" pos="{fmt(p_site_mov_b)}" quat="{fmtq(q_site_mov_b)}" size="0.003" rgba="0 1 0 0.6"/>')
    print(f'<geom name="moving_fingertip" type="sphere" size="{r:.6f}" pos="{fmt(p_sphere_mov_b)}" '
          f'class="task_objects" material="gripper_rubber" friction="1.0 0.003 0.0001" contype="0" conaffinity="2"/>')

    # 6) Optional verification in MuJoCo
    if args.verify:
        m = mujoco.MjModel.from_xml_path(args.xml)
        d = mujoco.MjData(m)

        # Insert the two spheres on-the-fly for evaluation (not modifying the file):
        # Instead of editing model structures (complicated), just report the world positions the
        # above BODY-frame spheres would land at, given the current kinematics.
        # To do that, we need world transforms of those bodies at the desired closed angle.
        # We'll find the BODIES by the GEOM search we already did.

        # Helper: find body id for an Element
        def body_id_by_elem(elem):
            # Find body "name" attribute by walking up until a body tag
            while elem is not None and elem.tag != "body":
                elem = elem.getparent()
            if elem is None:
                raise ValueError("Could not map geom to body element")
            body_name = elem.get("name")
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name), body_name

        bid_fix, bname_fix = body_id_by_elem(g_fix)
        bid_mov, bname_mov = body_id_by_elem(g_mov)

        # Set gripper closed angle
        try:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            d.qpos[jid] = float(args.closed_angle)
        except mujoco.MujocoError:
            pass
        mujoco.mj_forward(m, d)

        # WORLD transforms of those bodies
        R_fix_w = d.xmat[bid_fix].reshape(3,3)
        t_fix_w = d.xpos[bid_fix]
        R_mov_w = d.xmat[bid_mov].reshape(3,3)
        t_mov_w = d.xpos[bid_mov]

        p_fix_w = t_fix_w + R_fix_w @ p_sphere_fix_b
        p_mov_w = t_mov_w + R_mov_w @ p_sphere_mov_b
        dist = np.linalg.norm(p_fix_w - p_mov_w)

        print("\n=== Verify (world-space) ===")
        print(f"Body (fixed) : {bname_fix}")
        print(f"Body (moving): {bname_mov}")
        print(f"Closed angle:  {args.closed_angle:.4f} rad")
        print(f"Sphere radius: {r:.6f} m  →  target center distance ≈ {2*r:.6f} m")
        print(f"Computed center distance:   {dist:.6f} m")
        print(f"Fixed sphere world:  {p_fix_w}")
        print(f"Moving sphere world: {p_mov_w}")
        if abs(dist - 2*r) > 1e-3:
            print("Note: centers are not exactly 2*r apart. Consider flipping one normal sign or adjusting offsets ±ε along each normal until distance ≈ 2*r.")
        else:
            print("OK: centers ~2*r apart (just touching).")

if __name__ == "__main__":
    main()