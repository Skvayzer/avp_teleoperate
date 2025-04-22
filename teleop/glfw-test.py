import glfw
import mujoco
from pathlib import Path


glfw.init()
window = glfw.create_window(800, 600, "MuJoCo", None, None)
glfw.make_context_current(window)

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "empty.xml"


# Only after context is current
model = mujoco.MjModel.from_xml_path(_XML.as_posix())
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)  # gladLoadGL inside

while not glfw.window_should_close(window):
    glfw.poll_events()
    glfw.swap_buffers(window)

glfw.terminate()
