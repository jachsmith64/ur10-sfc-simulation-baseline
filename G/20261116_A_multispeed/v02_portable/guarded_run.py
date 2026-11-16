"""Run a reviewed offline entry point in its own process, rejecting device access."""

import importlib.abc
from pathlib import Path
import platform
import runpy
import sys


class RejectDeviceModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        lower = fullname.lower()
        if lower.startswith(("rtde_control", "rtde_receive", "rtde_io", "dashboard_client",
                             "mvcameracontrol", "hikrobot", "pypylon", "pyrealsense")):
            raise ImportError(f"Offline device module blocked: {fullname}")
        return None


def reject_network(event, args):
    if event.startswith("socket.") and event != "socket.gethostname":
        raise RuntimeError(f"Offline network access blocked: {event}")
    if event in ("os.system", "subprocess.Popen", "os.startfile"):
        raise RuntimeError(f"Offline process launch blocked: {event}")


def main():
    platform.uname()
    sys.meta_path.insert(0, RejectDeviceModules())
    sys.addaudithook(reject_network)
    try:
        import cv2
    except ImportError:
        pass
    else:
        video_capture = cv2.VideoCapture

        def files_only(source=None, *args, **kwargs):
            if not isinstance(source, (str, Path)) or not Path(source).is_file():
                raise RuntimeError("Offline VideoCapture requires an existing local file")
            return video_capture(str(source), *args, **kwargs)

        cv2.VideoCapture = files_only
    if sys.argv[1] == "-m":
        sys.argv = sys.argv[2:]
        sys.path.insert(0, str(Path.cwd()))
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    else:
        sys.argv = sys.argv[1:]
        script = Path(sys.argv[0]).resolve(strict=True)
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
