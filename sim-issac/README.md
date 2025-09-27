## XleRobot IK Extension for Isaac Sim

This folder contains a custom Isaac Sim UI extension that loads the XleRobot USD model and controls its two arms with Lula inverse kinematics (IK).

### What you get
- Load the USD robot onto a new stage and create two target frames in world: `/World/target_right`, `/World/target_left`.
- Build two Lula IK solvers (one per arm) using your URDF and YAML descriptors.
- Drive the `SingleArticulation` each physics step with IK actions.
- Robust logging to compare world vs base frames and debug IK convergence.

### Layout
- `sim-issac/xlerobot_ik_exts/`
  - `Lula_ik_python/extension.py`: Extension entrypoint and window/menu wiring
  - `Lula_ik_python/ui_builder.py`: UI (Load/Reset/Run), world lifecycle, scenario orchestration
  - `Lula_ik_python/scenario.py`: Core logic (load robot/targets, set up IK, per-step updates, debug)
  - `Lula_ik_python/right_arm.yaml`, `left_arm.yaml`: Lula robot descriptors for each arm
- Robot assets
  - `sim-issac/xlerobot_maniskill/xlerobot issac.usd`: Robot USD referenced by the extension
  - `sim-issac/xlerobot_maniskill/xlerobot issac.urdf`: URDF used by Lula solvers

### How the IK pipeline works
1. Targets live in world coordinates. Each frame we read target pose from `/World/target_right|left`.
2. For each arm, we set the solver’s base pose from the arm’s base link world pose:
   - Right: prim named `Base` under `/World/xlerobot/.../base_link/Base`
   - Left: prim named `Base_2` under `/World/xlerobot/.../base_link/Base_2`
   The extension discovers these nested paths automatically.
3. We call `ArticulationKinematicsSolver.compute_inverse_kinematics(target_world_pose)`.
   - Lula internally maps world → base using the provided base pose.
4. We apply the resulting joint targets to the articulation. By default we also apply “near solutions” even when the solver success flag is False to allow iterative convergence (`_apply_action_on_fail = True`).
5. For debugging, we also print:
   - Base link world pose (from wrapper and from stage USD)
   - End-effector pose and target in the same frame (world) so the errors are meaningful

Notes:
- `compute_end_effector_pose()` returns the EE pose in the solver’s base frame. The extension converts it to world for apples-to-apples logging against the world target.
- For 5‑DOF arms, full 6D (position + orientation) IK is often overconstrained; position-only or relaxed orientation increases robustness.

### Why the extension “just works” now
- Correct frames: we pass the true base link world pose per arm, not the articulation root.
- Correct assets: the USD is referenced and PhysX world is initialized (`World.reset()`), so articulation state is valid for IK.
- Robust prim lookup: articulation root and base links are found even when nested under reference composition.
- Iterative convergence: actions are applied even when the solver’s strict success check reports a near‑miss.

### Changes made to your URDF and descriptors
- URDF (`xlerobot issac.urdf`)
  - Set positive velocity limits for all movable joints (Lula rejects zero velocities).
  - Fixed base joints at the root to eliminate invalid inertia warnings and base motion.
  - Added tip convenience frames: `Fixed_Jaw_tip`, `Fixed_Jaw_tip_2` (fixed joints with rpy="0 0 0") for clean EE targeting.
  - Wrist→jaw joints use `rpy="0 1.5708 0"` (+90° pitch), which influences the EE world orientation when joints are at home.
- YAML (`right_arm.yaml`, `left_arm.yaml`)
  - Specify `root_link` (`Base` / `Base_2`).
  - Provide velocity/acceleration/jerk limits for stability.
  - Ensure cspace joint names/order match the URDF chain to the EE.

### Using the extension
1. Launch Isaac Sim and enable the extension from the menu (the window appears docked to the left).
2. Click “LOAD”
   - Creates a new `World`, new stage, references the USD at `/World/xlerobot`.
   - Creates `/World/target_right` and `/World/target_left` with default poses.
   - Calls `World.reset()` to initialize PhysX.
3. Click “RUN”
   - The extension subscribes to physics steps and executes IK.
   - Move the target prims in the viewport to command the arms.
4. Click “RESET” to reset physics and defaults.

### Configuration knobs (edit `scenario.py`)
- End-effectors: set `RIGHT_EE_NAME`, `LEFT_EE_NAME` (prefer tip frames).
- Apply actions on IK failure: `_apply_action_on_fail = True` (default). Set to `False` to only move when solver reports success.
- Target defaults: edit `set_default_state` positions/orientations in `load_example_assets()`.
- Debug frequency: `_debug_every_n` controls how often logs print.

### Troubleshooting
- “IK failed” but errors look small: Lula’s success thresholds can be tight. Options:
  - Keep applying near‑solutions (default behavior) so the seed converges.
  - Relax orientation (position-only test) by setting the target quaternion equal to the current EE quaternion each frame.
  - Move targets in small increments so the seed stays close.
- Target shows Euler 0,0,180 even when quaternion is [0,0,0,1]: that’s just one Euler decomposition of identity; quaternions q and −q represent the same rotation.
- Can’t find base links at `/World/xlerobot/Base(_2)`: the extension searches nested prims (e.g., `.../base_link/Base`). Check the console for the discovered paths.
- “Could not find or spawn an articulation”: verify the USD path and that the stage is valid.
- Joints not moving: ensure drives exist and `World.reset()` was called; confirm that the `action` names match the articulation DOF names.

### Frame conventions
- Targets are in world. We set the solver’s base pose (world) each step. Lula maps world → base internally.
- EE pose from `compute_end_effector_pose()` is in base; the extension converts to world for logging.

### FAQ
- Why 5‑DOF arms struggle with 6D IK?
  - There’s no redundancy to satisfy arbitrary orientation. Use position-only or reduce orientation weight.
- Why does the target look “near zero” in logs sometimes?
  - That was base‑frame EE vs world target mismatch in earlier logs; current logs compare in the same frame.

### Credits
Built on Isaac Sim UI/World APIs and Lula Kinematics. Some utility patterns adapted from the Franka example.


