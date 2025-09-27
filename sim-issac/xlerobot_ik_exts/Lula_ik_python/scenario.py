import os
from pathlib import Path

import numpy as np

import omni.usd
from pxr import UsdPhysics, UsdGeom, Gf

from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)


# End-effector prim names on your robot. Adjust if different on your USD/URDF.
RIGHT_EE_NAME = "Fixed_Jaw_tip"
LEFT_EE_NAME = "Fixed_Jaw_tip_2"

TARGET_DEFAULT_POS = np.array([0.45, 0, 0.95])
TARGET_DEFAULT_QUAT = np.array([0.7011,0,0,0.7011])


def _quat_to_rotmat(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        q: Quaternion as (x, y, z, w) or similar sequence.

    Returns:
        3x3 numpy array representing the rotation matrix.
    """
    try:
        # Try to unpack the quaternion directly
        x, y, z, w = q
    except Exception:
        # Fall back if given as tuple-like (e.g., numpy array)
        x, y, z, w = q[0], q[1], q[2], q[3]
    # Normalize the quaternion to avoid scaling issues
    n = x * x + y * y + z * z + w * w
    if n > 0.0:
        s = 2.0 / n
    else:
        s = 0.0
    # Precompute products for efficiency
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    # Construct the rotation matrix using the standard formula
    return np.array([
        [1.0 - (yy + zz),        xy - wz,        xz + wy],
        [       xy + wz, 1.0 - (xx + zz),        yz - wx],
        [       xz - wy,        yz + wx, 1.0 - (xx + yy)],
    ])


def _angle_between_rotmats(Ra, Rb):
    try:
        M = Ra.T @ Rb
        c = (np.trace(M) - 1.0) * 0.5
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.arccos(c))
    except Exception:
        return float("nan")


class XleRobotKinematicsScenario:
    def __init__(self):
        self._articulation = None
        self._target_right = None
        self._target_left = None
        self._base_right = None
        self._base_left = None
        self._base_right_path = None
        self._base_left_path = None
        # Control whether to apply IK actions even if solver reports failure
        self._apply_action_on_fail = True

        self._right_solver = None
        self._left_solver = None
        self._right_articulation_ik = None
        self._left_articulation_ik = None

        # Paths relative to this file
        self._pkg_dir = Path(__file__).resolve().parent
        sim_issac_dir = self._pkg_dir.parent.parent
        self._usd_path = str((sim_issac_dir / "xlerobot_maniskill" / "xlerobot issac" / "xlerobot issac.usd").resolve())
        self._urdf_path = str((sim_issac_dir / "xlerobot_maniskill" / "xlerobot issac.urdf").resolve())
        self._right_yaml = str((self._pkg_dir / "right_arm.yaml").resolve())
        self._left_yaml = str((self._pkg_dir / "left_arm.yaml").resolve())

        # Debug counters
        self._step_count = 0
        self._debug_every_n = 30

    # ----------------------- Stage / assets -----------------------
    def _find_articulations_in_stage(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []
        paths = []
        for prim in stage.Traverse():
            try:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    paths.append(prim.GetPath().pathString)
            except Exception:
                pass
        return paths

    def _find_articulation_under(self, anchor_path: str):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        anchor = stage.GetPrimAtPath(anchor_path)
        if not anchor.IsValid():
            return None
        # Check anchor itself first
        try:
            if anchor.HasAPI(UsdPhysics.ArticulationRootAPI):
                return anchor.GetPath().pathString
        except Exception:
            pass
        # Then search its subtree
        for prim in anchor.GetChildren():
            try:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    return prim.GetPath().pathString
            except Exception:
                pass
        # Fallback: broader subtree traversal
        anchor_prefix = anchor_path.rstrip("/") + "/"
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if not p.startswith(anchor_prefix):
                continue
            try:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    return p
            except Exception:
                pass
        return None

    # Removed composition waiting to avoid potential crashes; rely on post-reference scans

    def _find_prim_by_name_under(self, anchor_path: str, name: str):
        """Find the first prim with exact name under the given anchor path."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        anchor = stage.GetPrimAtPath(anchor_path)
        if not anchor or not anchor.IsValid():
            return None
        prefix = anchor_path.rstrip("/") + "/"
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if not p.startswith(prefix):
                continue
            if prim.GetName() == name:
                return p
        return None

    def load_example_assets(self):
        """Ensure robot USD exists on stage and create target frames.

        Returns tuple: (articulation, target_right, target_left)
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No stage active. Open or create a stage before loading assets.")

        # If no articulation present, reference the USD
        found_paths = self._find_articulations_in_stage()
        art_path = found_paths[0] if found_paths else None
        if art_path is None and os.path.isfile(self._usd_path):
            robot_prim_path = "/World/xlerobot"
            add_reference_to_stage(self._usd_path, robot_prim_path)
            # Search under the anchor for an articulation root
            art_path = self._find_articulation_under(robot_prim_path)
            if art_path is None:
                # Last resort: search entire stage
                found_paths = self._find_articulations_in_stage()
                art_path = found_paths[0] if found_paths else None

        if art_path is None:
            # Helpful debug printing
            try:
                anchor = stage.GetPrimAtPath("/World/xlerobot")
                print("[XleRobot] /World/xlerobot valid:", anchor.IsValid(), "schemas:", anchor.GetAppliedSchemas())
                print("[XleRobot] children:", [c.GetPath().pathString for c in anchor.GetChildren()])
            except Exception:
                pass
            raise RuntimeError("Could not find or spawn an articulation for xlerobot on the stage.")

        print(f"[XleRobot] Using articulation root: {art_path}")
        self._articulation = Articulation(art_path)

        # Create or get targets
        if stage.GetPrimAtPath("/World/target_right"):
            self._target_right = XFormPrim("/World/target_right", name="target_right")
        else:
            self._target_right = XFormPrim("/World/target_right", name="target_right", scale=[0.04, 0.04, 0.04])
        # Always set a sane default so world.reset() positions it correctly
        self._target_right.set_default_state(TARGET_DEFAULT_POS, TARGET_DEFAULT_QUAT)

        if stage.GetPrimAtPath("/World/target_left"):
            self._target_left = XFormPrim("/World/target_left", name="target_left")
        else:
            self._target_left = XFormPrim("/World/target_left", name="target_left", scale=[0.04, 0.04, 0.04])
        # Always set a sane default so world.reset() positions it correctly
        self._target_left.set_default_state(TARGET_DEFAULT_POS, TARGET_DEFAULT_QUAT)

        # Cache base link prims for correct base pose mapping per arm (search by name)
        self._base_right_path = self._find_prim_by_name_under("/World/xlerobot", "Base")
        self._base_left_path = self._find_prim_by_name_under("/World/xlerobot", "Base_2")
        if self._base_right_path:
            try:
                self._base_right = XFormPrim(self._base_right_path)
                print(f"[XleRobot] Found right base link at: {self._base_right_path}")
            except Exception:
                self._base_right = None
        else:
            print("[XleRobot] Right base link named 'Base' not found under /World/xlerobot")
        if self._base_left_path:
            try:
                self._base_left = XFormPrim(self._base_left_path)
                print(f"[XleRobot] Found left base link at: {self._base_left_path}")
            except Exception:
                self._base_left = None
        else:
            print("[XleRobot] Left base link named 'Base_2' not found under /World/xlerobot")

        return self._articulation, self._target_right, self._target_left

    # ----------------------- IK setup -----------------------
    def setup(self):
        if self._articulation is None:
            self.load_example_assets()

        # Debug articulation info
        try:
            dof_names = getattr(self._articulation, "dof_names", None)
            num_dof = getattr(self._articulation, "num_dof", None)
            print(f"[XleRobot] Articulation DOF count={num_dof} names={dof_names}")
        except Exception:
            pass

        # Right arm IK
        if os.path.isfile(self._right_yaml) and os.path.isfile(self._urdf_path):
            self._right_solver = LulaKinematicsSolver(
                robot_description_path=self._right_yaml,
                urdf_path=self._urdf_path,
            )
            self._right_articulation_ik = ArticulationKinematicsSolver(
                self._articulation,
                self._right_solver,
                RIGHT_EE_NAME,
            )
            try:
                print("[XleRobot][Right] Frames:", self._right_solver.get_all_frame_names())
            except Exception:
                pass

        # Left arm IK
        if os.path.isfile(self._left_yaml) and os.path.isfile(self._urdf_path):
            self._left_solver = LulaKinematicsSolver(
                robot_description_path=self._left_yaml,
                urdf_path=self._urdf_path,
            )
            self._left_articulation_ik = ArticulationKinematicsSolver(
                self._articulation,
                self._left_solver,
                LEFT_EE_NAME,
            )
            try:
                print("[XleRobot][Left] Frames:", self._left_solver.get_all_frame_names())
            except Exception:
                pass

    # ----------------------- Update per step -----------------------
    def update(self, step: float):
        if self._articulation is None:
            return

        # Wait until articulation is initialized by PhysX
        try:
            num_dof = self._articulation.num_dof
        except Exception:
            num_dof = 0
        if not num_dof:
            if (self._step_count % self._debug_every_n) == 0:
                print("[XleRobot] Articulation not initialized yet; waiting…")
            self._step_count += 1
            return

        base_pos, base_quat = self._articulation.get_world_pose()

        # Right arm
        if self._right_articulation_ik is not None and self._target_right is not None:
            # Use Base link pose for right arm if available
            try:
                if self._base_right is not None:
                    base_r_pos, base_r_quat = self._base_right.get_world_pose()
                else:
                    base_r_pos, base_r_quat = base_pos, base_quat
            except Exception:
                base_r_pos, base_r_quat = base_pos, base_quat
            self._right_solver.set_robot_base_pose(base_r_pos, base_r_quat)
            trg_pos, trg_quat = self._target_right.get_world_pose()
            if (self._step_count % self._debug_every_n) == 0:
                try:
                    bp = np.asarray(base_r_pos).tolist()
                    bq = np.asarray(base_r_quat).tolist()
                    print(f"[XleRobot][Right] Base prim={self._base_right_path or '[articulation root]'} frame=world pos={bp} quat={bq}")
                    # Also print stage-computed world transform for the base prim for comparison
                    stage = omni.usd.get_context().get_stage()
                    prim = stage.GetPrimAtPath(self._base_right_path) if self._base_right_path else None
                    if prim and prim.IsValid():
                        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
                        t = xf.ExtractTranslation()
                        rq = xf.ExtractRotation().GetQuat()
                        rq_list = [rq.GetReal(), rq.GetImaginary()[0], rq.GetImaginary()[1], rq.GetImaginary()[2]]
                        print(f"[XleRobot][Right] Stage world Base pos={[t[0], t[1], t[2]]} quat={rq_list}")
                    ee_pos_r, ee_rot_r = self._right_articulation_ik.compute_end_effector_pose()
                    pos_err_r = float(np.linalg.norm(np.array(trg_pos) - np.array(ee_pos_r)))
                    # Compute rough orientation error as angle between forward axes
                    R_t = _quat_to_rotmat(trg_quat)
                    R_e = ee_rot_r
                    # Use the Z axis (approach) as a proxy
                    z_t = R_t[:, 2]
                    z_e = R_e[:, 2]
                    dot = float(np.clip(np.dot(z_t, z_e), -1.0, 1.0))
                    ang = float(np.arccos(dot))
                    print(f"[XleRobot][Right] pre-IK target_pos={trg_pos} ee_pos={ee_pos_r} pos_err={pos_err_r:.4f} ori_err(rad)={ang:.3f}")
                except Exception as e:
                    print(f"[XleRobot][Right] pre-IK could not compute EE pose: {e}")
            # Extra debug: also compute base-frame target for comparison
            R_base_r = _quat_to_rotmat(base_r_quat)
            tgt_pos_r_b = R_base_r.T @ (np.array(trg_pos) - np.array(base_r_pos))
            # Note: articulation wrapper expects world-frame target while base pose is set
            action_r, ok_r = self._right_articulation_ik.compute_inverse_kinematics(trg_pos, trg_quat)
            if ok_r:
                try:
                    self._articulation.apply_action(action_r)
                except Exception:
                    pass
                if (self._step_count % self._debug_every_n) == 0:
                    print(f"[XleRobot][Right] IK ok. target={trg_pos}")
            else:
                # Apply near-solution anyway so the seed can converge across steps
                if self._apply_action_on_fail and action_r is not None:
                    try:
                        self._articulation.apply_action(action_r)
                        if (self._step_count % self._debug_every_n) == 0:
                            print(f"[XleRobot][Right] applying action despite IK failure")
                    except Exception:
                        if (self._step_count % self._debug_every_n) == 0:
                            print("[XleRobot][Right] IK failed and action could not be applied")
                if (self._step_count % self._debug_every_n) == 0:
                    try:
                        ee_pos_r_b, ee_rot_r_b = self._right_articulation_ik.compute_end_effector_pose()
                        pos_err_b = float(np.linalg.norm(tgt_pos_r_b - np.array(ee_pos_r_b)))
                        print(f"[XleRobot][Right] IK failed. world_pos_err={pos_err_r:.4f} base_pos_err={pos_err_b:.4f} tgt_b={tgt_pos_r_b.tolist()}")
                    except Exception:
                        print("[XleRobot][Right] IK failed target=", trg_pos)

        # Left arm
        if self._left_articulation_ik is not None and self._target_left is not None:
            # Use Base_2 link pose for left arm if available
            try:
                if self._base_left is not None:
                    base_l_pos, base_l_quat = self._base_left.get_world_pose()
                else:
                    base_l_pos, base_l_quat = base_pos, base_quat
            except Exception:
                base_l_pos, base_l_quat = base_pos, base_quat
            self._left_solver.set_robot_base_pose(base_l_pos, base_l_quat)
            trg_pos, trg_quat = self._target_left.get_world_pose()
            if (self._step_count % self._debug_every_n) == 0:
                try:
                    bp = np.asarray(base_l_pos).tolist()
                    bq = np.asarray(base_l_quat).tolist()
                    print(f"[XleRobot][Left]  Base prim={self._base_left_path or '[articulation root]'} frame=world pos={bp} quat={bq}")
                    # Also print stage-computed world transform for the left base prim for comparison
                    stage = omni.usd.get_context().get_stage()
                    prim = stage.GetPrimAtPath(self._base_left_path) if self._base_left_path else None
                    if prim and prim.IsValid():
                        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
                        t = xf.ExtractTranslation()
                        rq = xf.ExtractRotation().GetQuat()
                        rq_list = [rq.GetReal(), rq.GetImaginary()[0], rq.GetImaginary()[1], rq.GetImaginary()[2]]
                        print(f"[XleRobot][Left]  Stage world Base_2 pos={[t[0], t[1], t[2]]} quat={rq_list}")
                    ee_pos_l, ee_rot_l = self._left_articulation_ik.compute_end_effector_pose()
                    pos_err_l = float(np.linalg.norm(np.array(trg_pos) - np.array(ee_pos_l)))
                    R_tl = _quat_to_rotmat(trg_quat)
                    z_tl = R_tl[:, 2]
                    z_el = ee_rot_l[:, 2]
                    dotl = float(np.clip(np.dot(z_tl, z_el), -1.0, 1.0))
                    angl = float(np.arccos(dotl))
                    print(f"[XleRobot][Left] pre-IK target_pos={trg_pos} ee_pos={ee_pos_l} pos_err={pos_err_l:.4f} ori_err(rad)={angl:.3f}")
                except Exception as e:
                    print(f"[XleRobot][Left] pre-IK could not compute EE pose: {e}")
            R_base_l = _quat_to_rotmat(base_l_quat)
            tgt_pos_l_b = R_base_l.T @ (np.array(trg_pos) - np.array(base_l_pos))
            action_l, ok_l = self._left_articulation_ik.compute_inverse_kinematics(trg_pos, trg_quat)
            if ok_l:
                try:
                    self._articulation.apply_action(action_l)
                except Exception:
                    pass
                if (self._step_count % self._debug_every_n) == 0:
                    print(f"[XleRobot][Left] IK ok. target={trg_pos}")
            else:
                if self._apply_action_on_fail and action_l is not None:
                    try:
                        self._articulation.apply_action(action_l)
                        if (self._step_count % self._debug_every_n) == 0:
                            print(f"[XleRobot][Left]  applying action despite IK failure")
                    except Exception:
                        if (self._step_count % self._debug_every_n) == 0:
                            print("[XleRobot][Left]  IK failed and action could not be applied")
                if (self._step_count % self._debug_every_n) == 0:
                    try:
                        ee_pos_l_b, ee_rot_l_b = self._left_articulation_ik.compute_end_effector_pose()
                        pos_err_b = float(np.linalg.norm(tgt_pos_l_b - np.array(ee_pos_l_b)))
                        print(f"[XleRobot][Left]  IK failed. world_pos_err={pos_err_l:.4f} base_pos_err={pos_err_b:.4f} tgt_b={tgt_pos_l_b.tolist()}")
                    except Exception:
                        print("[XleRobot][Left]  IK failed target=", trg_pos)

        self._step_count += 1

    def reset(self):
        # IK is stateless; targets can be moved back to defaults by the caller if needed
        pass