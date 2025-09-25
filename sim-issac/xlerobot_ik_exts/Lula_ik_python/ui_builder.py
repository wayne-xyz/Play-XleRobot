import omni.timeline
import omni.ui as ui


class StageControlUI:
    def __init__(self):
        self._timeline = omni.timeline.get_timeline_interface()
        self._start_btn = None
        self._pause_btn = None
        self._stop_btn = None

    def _on_start(self):
        self._report_articulations()
        self._timeline.play()
        self._update_buttons()

    def _on_pause(self):
        self._timeline.pause()
        self._update_buttons()

    def _on_stop(self):
        self._timeline.stop()
        self._update_buttons()

    def _update_buttons(self):
        playing = self._timeline.is_playing() if self._timeline is not None else False
        if self._start_btn is not None:
            self._start_btn.enabled = not playing
        if self._pause_btn is not None:
            self._pause_btn.enabled = playing
        if self._stop_btn is not None:
            self._stop_btn.enabled = True

    def build_ui(self):
        with ui.VStack(spacing=8, height=0):
            ui.Label("Stage Controls", height=24)
            with ui.HStack(spacing=8, height=0):
                self._start_btn = ui.Button("Start", clicked_fn=self._on_start)
                self._pause_btn = ui.Button("Pause", clicked_fn=self._on_pause)
                self._stop_btn = ui.Button("Stop", clicked_fn=self._on_stop)
        self._update_buttons()

    def _report_articulations(self):
        try:
            import omni.usd
            from pxr import UsdPhysics
            from isaacsim.core.prims import SingleArticulation as Articulation
        except Exception as exc:
            print(f"[StageControls] Could not import runtime modules to report articulations: {exc}")
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[StageControls] No stage loaded.")
            return

        articulation_paths = []
        for prim in stage.Traverse():
            try:
                # Check if the prim has the ArticulationRoot API applied
                schemas = prim.GetAppliedSchemas()
                if schemas and "PhysicsArticulationRootAPI" in schemas:
                    articulation_paths.append(prim.GetPath().pathString)
            except Exception:
                pass

        if not articulation_paths:
            print("[StageControls] No articulations found on the current stage.")
            return

        print(f"[StageControls] Found {len(articulation_paths)} articulation(s):")
        for path in articulation_paths:
            try:
                art = Articulation(path)
                num_dof = getattr(art, "num_dof", None)
                dof_names = getattr(art, "dof_names", None)
                print(f"  - {path}")
                if num_dof is not None:
                    print(f"    DOF count: {num_dof}")
                if dof_names is not None:
                    try:
                        print(f"    Joint names: {', '.join(dof_names)}")
                    except Exception:
                        print(f"    Joint names: {dof_names}")
            except Exception as e:
                print(f"  - {path}: failed to query articulation info ({e})")

    def cleanup(self):
        self._start_btn = None
        self._pause_btn = None
        self._stop_btn = None


