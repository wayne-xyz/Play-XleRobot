import omni.ext
import omni.ui as ui

from .ui_builder import StageControlUI


class LulaIkMinimalExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._window = ui.Window("Stage Controls", width=300, height=120)
        self._ui = StageControlUI()
        with self._window.frame:
            self._ui.build_ui()

    def on_shutdown(self):
        if hasattr(self, "_ui") and self._ui is not None:
            self._ui.cleanup()
        if hasattr(self, "_window") and self._window is not None:
            self._window = None


