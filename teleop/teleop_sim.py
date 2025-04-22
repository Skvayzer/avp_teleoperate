import mujoco
import numpy as np
from pytransform3d import rotations

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices

from pathlib import Path
import time
import yaml
from multiprocessing import Process, shared_memory, Queue, Manager, Event, Lock

import cv2
import zmq
import pickle
import zlib
import socket
import glfw

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from robot_control.robot_hand import H1HandController
from teleop.robot_control.robot_arm import H1ArmController
from teleop.robot_control.robot_arm_ik import Arm_IK

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

import mujoco
import mujoco.viewer

from dex_retargeting.retargeting_config import RetargetingConfig


_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "empty.xml"


def send_data(server_socket, data):
    try:
        server_socket.send(data.encode('utf-8'))
    except ConnectionResetError as e:
        print(f"Connection reset by peer: {e}")
        # Attempt to reconnect
        server_socket.close()
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect(("server_address", server_port))
        server_socket.send(data.encode('utf-8'))


def image_receiver(image_queue, resolution, crop_size_w, crop_size_h):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.191:5555")

    while True:
        compressed_data = b''
        while True:
            chunk = socket.recv()
            compressed_data += chunk
            if len(chunk) < 60000:
                break
        data = zlib.decompress(compressed_data)
        frame_data = pickle.loads(data)

        # Decode and display the image
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sm.write_image(frame)
        # Control receiving frequency
        time.sleep(0.01)


class SharedMemoryImage:
    def __init__(self, img_shape):
        self.resolution = img_shape  # (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h, self.resolution[1] - 2 * self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        self.lock = Lock()

    def write_image(self, image):
        with self.lock:
            np.copyto(self.img_array, image)

    def read_image(self):
        with self.lock:
            image_copy = self.img_array.copy()
            return image_copy

    def cleanup(self):
        self.shm.close()
        self.shm.unlink()


class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (800, 640)  # (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h, self.resolution[1] - 2 * self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=False)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        head_rmat = head_mat[:3, :3]

        left_wrist_mat[2, 3] +=0.45
        right_wrist_mat[2,3] +=0.45
        left_wrist_mat[0, 3] +=0.20
        right_wrist_mat[0,3] +=0.20

        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7]]
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[]]

        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos
        # return left_wrist_mat, right_wrist_mat


    def get_arm_state(self):
        """Get current arm state from MuJoCo simulation"""
        # Get current joint positions for arms
        armstate = np.zeros(14)  # 7 joints per arm
        armv = np.zeros(14)      # 7 joint velocities per arm

        # Left arm joints (7 joints)
        # 15: left_shoulder_pitch_joint
        # 16: left_shoulder_roll_joint
        # 17: left_shoulder_yaw_joint
        # 18: left_elbow_joint
        # 19: left_hand_joint
        # 20: L_thumb_proximal_yaw_joint
        # 21: L_thumb_proximal_pitch_joint
        armstate[:7] = self.data.qpos[15:22]
        armv[:7] = self.data.qvel[15:22]

        # Right arm joints (7 joints)
        # 32: right_shoulder_pitch_joint
        # 33: right_shoulder_roll_joint
        # 34: right_shoulder_yaw_joint
        # 35: right_elbow_joint
        # 36: right_hand_joint
        # 37: R_thumb_proximal_yaw_joint
        # 38: R_thumb_proximal_pitch_joint
        armstate[7:14] = self.data.qpos[32:39]
        armv[7:14] = self.data.qvel[32:39]

        return armstate, armv


class MujocoVRRenderer:
    """
    MuJoCo renderer for VR teleoperation simulation.
    Renders stereo images for VR headset display.
    """

    def __init__(self, model, data, width=800, height=600):
        """
        Initialize the VR renderer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Width of each eye's viewport
            height: Height of each eye's viewport
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Create a window for rendering (can be hidden in production)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(width*2, height, "MuJoCo VR Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        # Make the window's context current
        glfw.make_context_current(self.window)

        # Initialize visualization data structures
        self.init_visualization()

        print(f"Initialized VR renderer with resolution {width*2}x{height}")

    def init_visualization(self):
        """Initialize visualization objects for MuJoCo."""
        # Create abstract visualization options
        self.vopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.vopt)

        # Create abstract camera
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)

        # Set camera type to USER - we'll control the cameras manually for VR
        self.cam.type = mujoco.mjtCamera.mjCAMERA_USER

        # Create scene with stereo settings
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE

        # Create rendering context
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Get framebuffer size for viewport
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # Create buffer for offscreen rendering
        self.img_buffer = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)

        # Set inter-pupillary distance (IPD) for stereo rendering
        self.ipd = 0.064  # 64mm - typical human average

        # Flag for first update
        self.first_update = True

    def update_cameras(self, head_rotation=None):
        """Update camera positions and orientations based on head tracking."""
        if head_rotation is not None:
            # Camera position (using a fixed position initially)
            cam_pos = np.array([0.0, 0.0, 1.5])  # Arbitrary position

            # Forward direction calculated from head rotation
            forward = head_rotation @ np.array([0, 0, -1])  # Default forward is -Z

            # Calculate right vector (cross product of forward and world up)
            world_up = np.array([0, 0, 1])
            right = np.cross(forward, world_up)
            right = right / np.linalg.norm(right)

            # Calculate camera up vector
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Set left eye camera
            left_pos = cam_pos - right * (self.ipd / 2)
            self.scn.camera[0].pos = left_pos
            self.scn.camera[0].forward = forward
            self.scn.camera[0].up = up

            # Set right eye camera
            right_pos = cam_pos + right * (self.ipd / 2)
            self.scn.camera[1].pos = right_pos
            self.scn.camera[1].forward = forward
            self.scn.camera[1].up = up

            # Only set frustum parameters on first update or if needed
            if self.first_update:
                # Set frustum for each eye
                for i in range(2):
                    self.scn.camera[i].frustum_near = 0.01
                    self.scn.camera[i].frustum_far = 100.0
                    self.scn.camera[i].frustum_center = 0.0
                    self.scn.camera[i].frustum_width = 0.1  # Controls FOV
                self.first_update = False

    def render_vr(self, head_rotation=None):
        """Render stereo images for VR display."""
        # Process window events
        glfw.poll_events()

        # Update cameras based on head rotation
        self.update_cameras(head_rotation)

        # Update scene with current model state
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn
        )

        # Check if context size needs updating
        if (self.viewport.width > self.ctx.offWidth or
            self.viewport.height > self.ctx.offHeight):
            self.ctx.free()
            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Render to window
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        glfw.swap_buffers(self.window)

        # Also render to offscreen buffer for VR streaming
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)

        # Read pixels from the offscreen buffer
        mujoco.mjr_readPixels(self.img_buffer, None, self.viewport, self.ctx)

        # Flip the image vertically (OpenGL renders from bottom to top)
        vr_image = np.flipud(self.img_buffer.copy())

        return vr_image

    def cleanup(self):
        """Clean up resources."""
        self.ctx.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == '__main__':
    # Initialize GLFW globally
    glfw.init()

    manager = Manager()
    image_queue = manager.Queue()
    teleoperator = VuerTeleop('inspire_hand.yml')

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Print model information
    print(f"Number of joints: {model.njnt}")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of actuators: {model.nu}")

    # Print joint names and ranges
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        range_min = model.jnt_range[i, 0]
        range_max = model.jnt_range[i, 1]
        print(f"Joint {i}, {name}: range [{range_min:.2f}, {range_max:.2f}]")

    # Initialize controllers and solvers
    h1hand = H1HandController()
    urdf_path = '../assets/h1/h1_with_hand.urdf'
    mesh_path = '../assets/h1/'
    arm_ik = Arm_IK(urdf_path, mesh_path)

    # Initialize renderer
    vr_renderer = MujocoVRRenderer(model, data,
                                  width=teleoperator.img_width,
                                  height=teleoperator.img_height)

    # Initialize shared memory for visualization
    sm = SharedMemoryImage((1600, 1280))

    # Reset MuJoCo data to initial state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    start_time = time.time()  # Capture script start time

    try:
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program): ")
        if user_input.lower() == 's':
            print("Starting teleoperation simulation...")

            # Main simulation loop
            while not glfw.window_should_close(vr_renderer.window):
                # Get handstate from controller
                handstate = h1hand.get_hand_state()

                # Process VR input - get head, hand poses and finger positions
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()

                # Process finger angles for control
                right_angles = [right_qpos[i] for i in [0, 2, 6, 4]]
                right_angles.append(right_qpos[8])
                right_angles.append(right_qpos[9])

                left_angles = [left_qpos[i] for i in [0, 2, 6, 4]]
                left_angles.append(left_qpos[8])
                left_angles.append(left_qpos[9])

                # Run IK solver
                sol_q, tau_ff, flag = arm_ik.ik_fun(
                    left_pose,
                    right_pose,
                    motorstate=None,
                    motorV=None,
                    left_hand_angles=left_angles,
                    right_hand_angles=right_angles
                )

                # Prepare control values
                sol_q_values = sol_q[-8:].tolist()  # Get last 8 values
                sol_q_values.append(0)  # Append zero
                tau_ff_values = tau_ff[-8:].tolist()
                tau_ff_values.append(0)

                # # Control the robot
                # for i in range(model.nu):  # model.nu is the number of actuators
                #     actuator_name = data.actuator(i).name

                #     try:
                #         correspond_joint = data.joint(actuator_name)
                #     except KeyError:
                #         continue
                #     correspond_joint_id = correspond_joint.id

                #     # Handle arm joints (11-18)
                #     if i >= 11 and i <= 18:
                #         data.ctrl[i] = (
                #             0  # feedforward term tau
                #             + 800 * (sol_q_values[i - 11] - data.qpos[correspond_joint_id])  # position control kp
                #             + 20 * (tau_ff_values[i - 11] - data.qvel[correspond_joint_id])  # velocity control kd
                #         )
                #     # Handle hand fingers (19-42)
                #     elif i >= 19 and i <= 42:
                #         target_angle = None

                #         if i <= 30:  # Left hand
                #             angles = left_angles
                #             prefix = "l_"
                #         else:  # Right hand
                #             angles = right_angles
                #             prefix = "r_"

                #         if f"{prefix}thumb" in actuator_name:
                #             if "proximal_1" in actuator_name:
                #                 target_angle = angles[4]  # thumb yaw
                #             elif "proximal_2" in actuator_name:
                #                 target_angle = angles[5]  # thumb pitch
                #         elif "proximal" in actuator_name:
                #             if f"{prefix}index" in actuator_name:
                #                 target_angle = angles[0]
                #             elif f"{prefix}middle" in actuator_name:
                #                 target_angle = angles[1]
                #             elif f"{prefix}ring" in actuator_name:
                #                 target_angle = angles[2]
                #             elif f"{prefix}pinky" in actuator_name:
                #                 target_angle = angles[3]

                #         if target_angle is not None:
                #             data.ctrl[i] = (
                #                 0  # feedforward term tau
                #                 + 1000 * (target_angle - data.qpos[correspond_joint_id])  # position control kp
                #                 + 40 * (1 - data.qvel[correspond_joint_id])  # velocity control kd
                #             )

                # # Step the simulation
                # mujoco.mj_step(model, data)

                # Render the scene for VR
                vr_image = vr_renderer.render_vr(head_rotation=head_rmat)

                # Send the rendered image to the teleoperator and shared memory
                np.copyto(teleoperator.img_array, vr_image)
                # sm.write_image(vr_image)

                # Rate limit to prevent maxing out CPU
                time.sleep(0.01)

                # Print FPS every 100 frames
                if int(time.time() - start_time) % 5 == 0:
                    print(f"FPS: {1/(time.time() - start_time):.1f}")
                    start_time = time.time()

    except KeyboardInterrupt:
        print("Shutting down teleoperator...")
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        vr_renderer.cleanup()
        sm.cleanup()
        print("Cleanup complete. Exiting.")
