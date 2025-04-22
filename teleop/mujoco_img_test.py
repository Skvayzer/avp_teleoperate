import mujoco
import glfw
import numpy as np
import imageio
from pathlib import Path

# Locate the XML model relative to this script
_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "empty.xml"

# Single offscreen render and save
def main():
    # Initialize GLFW offscreen
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    glfw.window_hint(glfw.VISIBLE, 0)
    width, height = 800, 600
    window = glfw.create_window(width, height, "OffscreenRender", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create window for offscreen render")
    glfw.make_context_current(window)

    # Load model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Setup renderer
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    scn = mujoco.MjvScene(model, maxgeom=10000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Render one offscreen frame
    mujoco.mjv_updateScene(model, data, vopt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
    mujoco.mjr_render(viewport, scn, ctx)

    # Read pixels and flip
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img, None, viewport, ctx)
    img = np.flipud(img)

    # Save image
    out_file = _HERE / "mujoco_test_image.png"
    imageio.imwrite(str(out_file), img)
    print(f"Saved image to {out_file}")

    # Cleanup
    ctx.free()
    glfw.terminate()

if __name__ == "__main__":
    main() 