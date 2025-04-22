import mujoco
import numpy as np
import cv2
from pathlib import Path
import glfw

# Path to the XML file
_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "empty.xml"

def test_mujoco_render():
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # Reset data and run forward to get to a valid state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Set up rendering
    width, height = 800, 600
    
    # Create window (can be hidden)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(width, height, "MuJoCo Test Render", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window")
    
    # Make context current
    glfw.make_context_current(window)
    
    # Create visualization objects
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    
    scn = mujoco.MjvScene(model, maxgeom=10000)
    
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Get framebuffer size
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
    
    # Create image buffer
    img_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Update scene and render
    mujoco.mjv_updateScene(
        model, 
        data, 
        vopt, 
        None, 
        cam, 
        mujoco.mjtCatBit.mjCAT_ALL, 
        scn
    )
    
    # Render to window
    mujoco.mjr_render(viewport, scn, ctx)
    
    # Also render to offscreen buffer
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
    mujoco.mjr_render(viewport, scn, ctx)
    
    # Read pixels from the offscreen buffer
    mujoco.mjr_readPixels(img_buffer, None, viewport, ctx)
    
    # Flip the image vertically (OpenGL renders from bottom to top)
    rendered_image = np.flipud(img_buffer.copy())
    
    # Save the image
    cv2.imwrite("mujoco_render_test.png", cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    print("Image saved as mujoco_render_test.png")
    
    # Clean up resources
    ctx.free()
    glfw.destroy_window(window)
    glfw.terminate()
    
    return rendered_image

if __name__ == "__main__":
    test_mujoco_render()