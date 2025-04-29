import mujoco
import numpy as np
import cv2
from pathlib import Path
import glfw
# import time # No longer needed unless used elsewhere

# Path to the XML file
_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "empty.xml"

def test_mujoco_render():
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    # Set up rendering dimensions
    width, height = 800, 600

    # Create a hidden window (we only need the context)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.FALSE) # Prevent HiDPI scaling issues
    window = glfw.create_window(width, height, "MuJoCo Offscreen Render", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window")

    # Make the window's context current - Essential for MuJoCo rendering
    glfw.make_context_current(window)
    # glfw.swap_interval(0) # Optional: disable vsync if needed

    # Load MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Reset data to a valid state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Create visualization objects
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    # Configure camera (Front View)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[0] = 0.0
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.5
    cam.distance = 3.0
    cam.azimuth = 0.0   # Front view
    cam.elevation = -10.0

    scn = mujoco.MjvScene(model, maxgeom=10000)
    # Note: Scene flags like SHADOW, REFLECTION, SKYBOX, HAZE can be set here if desired
    # scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    # scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
    # scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    # scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = True # For fog background

    # Initialize MuJoCo rendering context
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Define the rendering viewport dimensions
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Create the NumPy buffer to hold the image
    img_buffer = np.zeros((height, width, 3), dtype=np.uint8)

    # Update the scene with the current model state
    mujoco.mjv_updateScene(
        model,
        data,
        vopt,
        None,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scn
    )

    # --- Offscreen Rendering Pipeline ---
    # 1. Select the offscreen buffer as the rendering target
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

    # 2. Render the scene to the selected (offscreen) buffer
    mujoco.mjr_render(viewport, scn, ctx)

    # 3. Read the pixels from the rendered (offscreen) buffer into the NumPy array
    #    The viewport specifies the region to read from the buffer.
    mujoco.mjr_readPixels(img_buffer, None, viewport, ctx)
    # -----------------------------------

    # Flip the image vertically because OpenGL renders bottom-up
    rendered_image = np.flipud(img_buffer)

    # Save the final image
    output_path = "mujoco_render_test.png"
    cv2.imwrite(output_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    print(f"Image saved as {output_path}")

    # Clean up resources
    ctx.free()
    glfw.destroy_window(window)
    glfw.terminate()

    return rendered_image

if __name__ == "__main__":
    # test_mujoco_render()
    model_mj = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data_mj = mujoco.MjData(model_mj)
    renderer = mujoco.Renderer(model_mj, 400, 640)
    mujoco.mj_forward(model_mj, data_mj)
    # Use a camera that exists in the model or use default camera
    # The 'fixed' camera doesn't exist in the empty.xml model
    renderer.update_scene(data_mj)  # Use default camera
    rendered_image = renderer.render()
    
    # Save the rendered image to file
    output_path = "mujoco_renderer_test.png"
    cv2.imwrite(output_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    print(f"Image saved as {output_path}")
