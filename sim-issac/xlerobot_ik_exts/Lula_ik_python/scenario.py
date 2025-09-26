import os
from pathlib import Path

import numpy as np

import omni.usd
from pxr import UsdPhysics

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


class XleRobotKinematicsScenario:
    def __init__(self):
        self._articulation = None
        self._target_right = None
        self._target_left = None

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
            self._target_right = XFormPrim("/World/target_right")
        else:
            self._target_right = XFormPrim("/World/target_right", scale=[0.04, 0.04, 0.04])
            self._target_right.set_default_state(np.array([0.45, -0.15, 0.95]), np.array([0, 0, 0, 1]))

        if stage.GetPrimAtPath("/World/target_left"):
            self._target_left = XFormPrim("/World/target_left")
        else:
            self._target_left = XFormPrim("/World/target_left", scale=[0.04, 0.04, 0.04])
            self._target_left.set_default_state(np.array([0.45, 0.15, 0.95]), np.array([0, 0, 0, 1]))

        return self._articulation, self._target_right, self._target_left

    # ----------------------- IK setup -----------------------
    def setup(self):
        if self._articulation is None:
            self.load_example_assets()

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

    # ----------------------- Update per step -----------------------
    def update(self, step: float):
        if self._articulation is None:
            return

        base_pos, base_quat = self._articulation.get_world_pose()

        # Right arm
        if self._right_articulation_ik is not None and self._target_right is not None:
            self._right_solver.set_robot_base_pose(base_pos, base_quat)
            trg_pos, trg_quat = self._target_right.get_world_pose()
            action_r, ok_r = self._right_articulation_ik.compute_inverse_kinematics(trg_pos, trg_quat)
            if ok_r:
                self._articulation.apply_action(action_r)

        # Left arm
        if self._left_articulation_ik is not None and self._target_left is not None:
            self._left_solver.set_robot_base_pose(base_pos, base_quat)
            trg_pos, trg_quat = self._target_left.get_world_pose()
            action_l, ok_l = self._left_articulation_ik.compute_inverse_kinematics(trg_pos, trg_quat)
            if ok_l:
                self._articulation.apply_action(action_l)

    def reset(self):
        # IK is stateless; targets can be moved back to defaults by the caller if needed
        pass