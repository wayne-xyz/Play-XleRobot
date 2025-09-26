import omni.timeline
import omni.ui as ui
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.style import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .scenario import XleRobotKinematicsScenario


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []

        self._timeline = omni.timeline.get_timeline_interface()
        self._is_loading = False
        self._on_init()

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            # Ignore stage-open events triggered by our own LOAD workflow
            if not self._is_loading:
                self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(spacing=8, height=0):
                    self._load_btn = ui.Button("LOAD", clicked_fn=self._on_click_load)
                    self._reset_btn = ui.Button("RESET", clicked_fn=self._on_click_reset, enabled=False)

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

    def _on_init(self):
        self._articulation = None
        self._target_right = None
        self._target_left = None
        self._scenario = XleRobotKinematicsScenario()

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        create_new_stage()
        self._add_light_to_stage()
        set_camera_view(eye=[1.5, 1.25, 2], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        loaded_objects = self._scenario.load_example_assets()

        world = World.instance()
        for loaded_object in loaded_objects:
            world.scene.add(loaded_object)

    def _setup_scenario(self):
        self._scenario.setup()

        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        self._scenario.reset()

        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        self._scenario.update(step)

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

    # ----------------- Button handlers -----------------
    def _on_click_load(self):
        # Create world/stage and load assets, then set up IK
        self._is_loading = True
        try:
            # Start with a fresh World so scene names are unique
            try:
                World.clear_instance()
            except Exception:
                pass
            world = World(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)

            create_new_stage()
            self._add_light_to_stage()
            set_camera_view(eye=[1.5, 1.25, 2], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
            loaded_objects = self._scenario.load_example_assets()
            try:
                articulation_obj, target_right_obj, target_left_obj = loaded_objects
            except Exception:
                articulation_obj = loaded_objects[0] if len(loaded_objects) > 0 else None
                target_right_obj = loaded_objects[1] if len(loaded_objects) > 1 else None
                target_left_obj = loaded_objects[2] if len(loaded_objects) > 2 else None

            if articulation_obj is not None:
                try:
                    world.scene.add(articulation_obj)
                except Exception as e:
                    print(f"[XleRobot] Skipping add articulation: {e}")
            if target_right_obj is not None:
                try:
                    world.scene.add(target_right_obj)
                except Exception as e:
                    print(f"[XleRobot] Skipping add target_right: {e}")
            if target_left_obj is not None:
                try:
                    world.scene.add(target_left_obj)
                except Exception as e:
                    print(f"[XleRobot] Skipping add target_left: {e}")
            self._setup_scenario()
        finally:
            self._is_loading = False

    def _on_click_reset(self):
        self._on_post_reset_btn()
