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

import os
import sys
import glfw
from OpenGL.GL import *
from scipy.spatial.transform import Rotation as R



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


# class VuerTeleop:
#     def __init__(self, config_file_path):
#         self.resolution = (600, 800)  # (720, 1280)
#         self.crop_size_w = 0
#         self.crop_size_h = 0
#         self.resolution_cropped = (self.resolution[0] - self.crop_size_h, self.resolution[1] - 2 * self.crop_size_w)

#         self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
#         self.img_height, self.img_width = self.resolution_cropped[:2]

#         self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
#         self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
#         image_queue = Queue()
#         toggle_streaming = Event()
#         self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=False)
#         self.processor = VuerPreprocessor()

#         RetargetingConfig.set_default_urdf_dir('../assets')
#         with Path(config_file_path).open('r') as f:
#             cfg = yaml.safe_load(f)
#         left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
#         right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
#         self.left_retargeting = left_retargeting_config.build()
#         self.right_retargeting = right_retargeting_config.build()

#     def step(self):
#         head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
#         head_rmat = head_mat[:3, :3]

#         left_wrist_mat[2, 3] +=0.45
#         right_wrist_mat[2,3] +=0.45
#         left_wrist_mat[0, 3] +=0.20
#         right_wrist_mat[0,3] +=0.20

#         left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
#         # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7]]
#         # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[]]
#         right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
#         # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[]]

#         return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos
#         # return left_wrist_mat, right_wrist_mat


#     def get_arm_state(self):
#         """Get current arm state from MuJoCo simulation"""
#         # Get current joint positions for arms
#         armstate = np.zeros(14)  # 7 joints per arm
#         armv = np.zeros(14)      # 7 joint velocities per arm
        
#         # Left arm joints (7 joints)
#         # 15: left_shoulder_pitch_joint
#         # 16: left_shoulder_roll_joint
#         # 17: left_shoulder_yaw_joint
#         # 18: left_elbow_joint
#         # 19: left_hand_joint
#         # 20: L_thumb_proximal_yaw_joint
#         # 21: L_thumb_proximal_pitch_joint
#         armstate[:7] = self.data.qpos[15:22]
#         armv[:7] = self.data.qvel[15:22]
        
#         # Right arm joints (7 joints)
#         # 32: right_shoulder_pitch_joint
#         # 33: right_shoulder_roll_joint
#         # 34: right_shoulder_yaw_joint
#         # 35: right_elbow_joint
#         # 36: right_hand_joint
#         # 37: R_thumb_proximal_yaw_joint
#         # 38: R_thumb_proximal_pitch_joint
#         armstate[7:14] = self.data.qpos[32:39]
#         armv[7:14] = self.data.qvel[32:39]
        
#         return armstate, armv



class VuerTeleop:
    def __init__(self, config_file_path):
        # Define base resolution (adjust as needed)
        self.resolution = (600, 800) # Example: Height, Width
        self.crop_size_w = 0 # Keep crop logic if needed, otherwise remove
        self.crop_size_h = 0
        # Resolution after cropping (if any)
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h, self.resolution[1] - 2 * self.crop_size_w)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        # Image shape for a single view
        self.img_shape = (self.img_height, self.img_width*2, 3)

        # Setup shared memory for the single image
        try:
            # Attempt to create shared memory
            # Use a unique name if multiple instances might run
            self.shm_name = f"teleop_sim_shm_{os.getpid()}"
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
            print(f"Created shared memory: {self.shm_name}")
        except FileExistsError:
             # If it already exists (e.g., from a previous run), link to it
             print(f"Shared memory block {self.shm_name} already exists. Linking...")
             # This path might indicate an issue if SHM wasn't cleaned up properly
             # Consider adding cleanup logic or ensuring unique names
             try:
                 self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
             except FileNotFoundError:
                 print(f"ERROR: SHM {self.shm_name} exists but couldn't be opened. Creating new.")
                 # Fallback to creating with potentially modified name or error handling
                 self.shm = shared_memory.SharedMemory(name=self.shm_name + "_fallback", create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)


        self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)



        # Initialize OpenTeleVision - Pass exact img_shape, dtype, and the event
        image_queue = Queue() # Keep if OpenTeleVision uses it
        toggle_streaming = Event() # Keep if OpenTeleVision uses it
        # Pass self.img_shape and np.uint8 (or appropriate dtype)
        self.tv = OpenTeleVision(self.img_shape, self.shm.name, image_queue, toggle_streaming, ngrok=False)

        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

        print(f"VuerTeleop initialized with image shape: {self.img_shape}")
        print(f"Shared memory size: {self.shm.size} bytes, Name: {self.shm.name}")

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        head_rmat = head_mat[:3, :3]

        # Keep coordinate adjustments if they are still correct
        left_wrist_mat[2, 3] +=0.45
        right_wrist_mat[2,3] +=0.45
        left_wrist_mat[0, 3] +=0.20
        right_wrist_mat[0,3] +=0.20

        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7]] # Preserved comment
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[]] # Preserved comment
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[]] # Preserved comment

        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos
        # return left_wrist_mat, right_wrist_mat # Preserved comment

    # Preserving commented out get_arm_state method
    # def get_arm_state(self):
    #     """Get current arm state from MuJoCo simulation"""
    #     # Get current joint positions for arms
    #     armstate = np.zeros(14)  # 7 joints per arm
    #     armv = np.zeros(14)      # 7 joint velocities per arm
    #
    #     # Left arm joints (7 joints)
    #     # 15: left_shoulder_pitch_joint
    #     # ... (other joint comments)
    #     armstate[:7] = self.data.qpos[15:22]
    #     armv[:7] = self.data.qvel[15:22]
    #
    #     # Right arm joints (7 joints)
    #     # 32: right_shoulder_pitch_joint
    #     # ... (other joint comments)
    #     armstate[7:14] = self.data.qpos[32:39]
    #     armv[7:14] = self.data.qvel[32:39]
    #
    #     return armstate, armv

    def cleanup(self):
        """Clean up shared memory."""
        if hasattr(self, 'shm') and self.shm:
            print(f"Cleaning up VuerTeleop shared memory: {self.shm.name}")
            self.shm.close()
            try:
                self.shm.unlink() # Attempt to unlink
                print(f"Shared memory {self.shm.name} unlinked.")
            except FileNotFoundError:
                print(f"Shared memory {self.shm.name} already unlinked or could not be found.")
        # Add cleanup for OpenTeleVision if needed
        # self.tv.cleanup()



if __name__ == '__main__':
    manager = Manager()
    image_queue = manager.Queue()
    teleoperator = VuerTeleop('inspire_hand.yml')

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # mujoco loading
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

    h1hand = H1HandController()
    # h1arm = H1ArmController()
    
    # Initialize IK solver with path to URDF and mesh files
    urdf_path = '../assets/h1/h1_with_hand.urdf'
    mesh_path = '../assets/h1/'
    arm_ik = Arm_IK(urdf_path, mesh_path)
    
    # sm = SharedMemoryImage((800, 640))
    # image_process = Process(target=image_receiver, args=(sm, teleoperator.resolution, teleoperator.crop_size_w, teleoperator.crop_size_h))
    # image_process.start()

    # --- Initialize GLFW and Create Display Window FIRST ---
    window = None # Initialize window variable
    texture_id = None # Initialize texture_id variable
    try:
        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        # Create a windowed mode window and its OpenGL context
        window_width, window_height = teleoperator.img_width*2, teleoperator.img_height
        window = glfw.create_window(window_width, window_height, "Teleop Sim Render (GLFW)", None, None)
        if not window:
            raise Exception("GLFW window can't be created")

        # Make the window's context current IMMEDIATELY
        glfw.make_context_current(window)
        print("GLFW window created and context is current.")

        # --- Setup OpenGL Texture (Requires window context to be current) ---
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0) # Unbind
        print("OpenGL texture initialized.")
        # -----------------------------------------------------------------

    except Exception as e:
        print(f"Error during GLFW/OpenGL setup: {e}")
        if window:
            glfw.destroy_window(window)
        glfw.terminate()
        sys.exit(1)
    # ---------------------------------------------------------

    # --- Initialize MuJoCo Renderer AFTER GLFW setup ---
    renderer = None # Initialize renderer variable
    try:
        print(f"Initializing mujoco.Renderer with Height: {teleoperator.img_height}, Width: {teleoperator.img_width}")
        # This might use the current context or create its own
        renderer = mujoco.Renderer(model, height=teleoperator.img_height, width=teleoperator.img_width)
    except Exception as e:
        print(f"Error initializing MuJoCo Renderer: {e}")
        if 'texture_id' in locals() and texture_id is not None:
             # Need context current to delete texture
             glfw.make_context_current(window)
             glDeleteTextures(1, [texture_id])
        if window:
            glfw.destroy_window(window)
        glfw.terminate()
        sys.exit(1)

    cam = mujoco.MjvCamera()
    # mujoco.mjv_defaultFreeCamera(model, cam) # Start with default free camera settings
    # cam.lookat = np.array([0.0, 0.0, 0.5]) # Look at robot base
    # cam.distance = 2.5
    # cam.elevation = -15.0 # S
    # Use fixed camera type to attach to robot's head
    # cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # Find the head/camera body ID in the model
    head_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera") 
    print(f"Head body ID: {head_cam_id}")
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = head_cam_id



    start_time = time.time()  # Capture script start time
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_address = ('192.168.123.162', 8080)
    # server_socket.connect(server_address)

    with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=True, show_right_ui=True
    ) as viewer:

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)

        try:
            user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
            if user_input.lower() == 's':
                # Setup ZMQ publisher
                # context = zmq.Context()
                # socket_init = False
                # socket = None
                # socket = context.socket(zmq.PUB)
                # socket.bind("tcp://*:8080")  # Change the port if needed
                # socket.connect("tcp://192.168.123.162:8080")  # Uncomment and comment above line if running h1_joint on h1 PC2
                step_counter = 0  # Initialize step counter

                while viewer.is_running():
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    armstate = None
                    armv = None
                    # frame = sm.read_image()
                    # np.copyto(teleoperator.img_array, np.array(frame))
                    handstate = h1hand.get_hand_state()

                    q_poseList = np.zeros(20)
                    q_tau_ff = np.zeros(20)

                    #l ock torso joint
                    data.qpos[10] = 0
                    # armstate,armv = h1arm.GetMotorState()

                    # hands uncomment# 
                    head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
                    # # print pose as quaternion
                    # print(rotations.quaternion_from_matrix(right_pose[:3,:3]))
                    # print(rotations.quaternion_from_matrix(left_pose[:3,:3]))
                    # # print pose translation
                    print(right_qpos, right_qpos.shape)
                    # print(left_pose[:3,3])



                    # --- Update Camera based on Head Rotation ---
                    # cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    # Assuming head_rmat is the orientation relative to a base frame
                    # Convert rotation matrix to Euler angles for azimuth/elevation control
                    try:
                        # Using pytransform3d assuming ZYX extrinsic ('sxyz')
                        # Verify this convention matches head_rmat source!
                        # Use the correct function name: matrix_to_extrinsic_euler_xyz
                        # euler_angles = rotations.matrix_to_extrinsic_euler_xyz(head_rmat) # Previous attempt
                        # Specify axes (0=X, 1=Y, 2=Z) and extrinsic=True for 'sxyz' convention
                        # euler_angles = rotations.euler_from_matrix(head_rmat, i=0, j=1, k=2, extrinsic=True)

                        # Map Euler angles: Yaw -> Azimuth, Pitch -> Elevation (negated)
                        # Ensure mapping is correct for your setup
                        # cam.azimuth = np.degrees(euler_angles[2])   # Z rotation (Yaw)
                        # cam.elevation = -np.degrees(euler_angles[1]) # Y rotation (Pitch, inverted)
                        # Roll (euler_angles[0]) is typically ignored for free camera orientation

                        # Create correction rotation: -90° around X axis
                        # correction_matrix = R.from_euler('x', -90, degrees=True).as_matrix() #@ R.from_euler('y', -45, degrees=True).as_matrix() @ R.from_euler('z', -45, degrees=True).as_matrix()
                        head_rotation = R.from_matrix(head_rmat)
                        # print("head_rotation: ", head_rotation)
                        yaw_pitch_roll = head_rotation.as_euler('yzx', degrees=False)  # axes: Yaw (Y) -> Pitch (X) -> Roll (Z)
                        print("yaw_pitch_roll: ", yaw_pitch_roll)
                        # print(head_rmat)
                        # Apply correction
                        # corrected_rmat = R.from_euler('x', -90, degrees=True).as_matrix() @ head_rmat @ R.from_euler('z', -90, degrees=True).as_matrix()
                        # col_exchange_matrix = np.array([
                        #     [0, 1,  0],
                        #     [1, 0,  0],
                        #     [0, 0,  1]
                        # ])


                        # print("raw quaternion: ", R.from_matrix(head_rmat).as_quat())
                        # # # Apply correction
                        # # corrected_rmat = correction_matrix @ head_rmat #@ col_exchange_matrix
                        # flip_Z = np.diag([1, 1, -1])
                        # corrected_rmat = head_rmat @ flip_Z

                        yaw = yaw_pitch_roll[0]
                        pitch = yaw_pitch_roll[1] #roll
                        roll = yaw_pitch_roll[2]

                        adjusted_yaw = -yaw  # when I pitch it yaws???
                        adjusted_pitch =  pitch#when I roll head it is pitch
                        adjusted_roll = roll #when I yaw it yaws in opposite direction


                        corrected_rotation = R.from_euler('xzy', [np.pi/2 + adjusted_yaw, -np.pi/2 + adjusted_pitch, adjusted_roll], degrees=False)
                        # print("corrected_rotation: ", corrected_rotation, corrected_rotation.as_matrix())

                        quat = corrected_rotation.as_quat()
                        # print("quat: ", quat)
                        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # reorder to [w, x, y, z] for MuJoCo
                        model.cam_quat[head_cam_id] = quat

                    except ValueError as e:
                        # Handle potential issues with matrix conversion (e.g., gimbal lock)
                        print(f"WARN: Could not convert head_rmat to Euler angles: {e}. Using previous camera settings.")

                    # # Fix the urdf order mismatch between the hand and the arm
                    # # import ipdb; ipdb.set_trace()
                    # original_armstate = armstate
                    # armstate = np.concatenate((armstate[4:8], armstate[0:4]))
                    # armv = np.concatenate((armv[4:8], armv[0:4]))

                    # left_pose, right_pose = teleoperator.step()

                    # 4,5: index 6,7: middle, 0,1: pinky, 2,3: ring, 8,9: thumb
                    right_angles = [right_qpos[i] for i in [0, 2, 6, 4]] #[4, 6, 2, 0]]
                    right_angles.append(right_qpos[8])
                    right_angles.append(right_qpos[9])
                
                    left_angles = [left_qpos[i] for i in  [0, 2, 6, 4]] #[4, 6, 2, 0]]
                    left_angles.append(left_qpos[8])
                    left_angles.append(left_qpos[9])

                    # Pass hand angles to IK solver for visualization
                    sol_q, tau_ff, flag = arm_ik.ik_fun(
                        left_pose, 
                        right_pose, 
                        motorstate=None, 
                        motorV=None,
                        left_hand_angles=left_angles,
                        right_hand_angles=right_angles
                    )

                    # Extract last 8 values, append a zero, format as space-separated string
                    sol_q_values = sol_q[10:14].tolist() + sol_q[27:31].tolist()
                    # sol_q_values.append(0)  # Append zero
                    # print(sol_q_values)

                    tau_ff_values = tau_ff[10:14].tolist() + tau_ff[27:31].tolist()
                    # tau_ff_values.append(0)  # Append zero
                    # print(tau_ff_values)

                    # Combine into a single space-separated string
                    combined_data = " ".join(map(str, sol_q_values + tau_ff_values))
                    # send_data(server_socket, combined_data)

                    # sol_q = np.concatenate((sol_q[4:8], sol_q[0:4]))
                    # tau_ff = np.concatenate((tau_ff[4:8], tau_ff[0:4]))

                    # if flag:
                    #     q_poseList[12:20] = sol_q
                    #     q_tau_ff[12:20] = tau_ff
                    # else:
                    #     q_poseList[12:20] = original_armstate
                    #     q_tau_ff = np.zeros(20)

                    # h1arm.SetMotorPose(q_poseList, q_tau_ff)
                    # print('Poselist:', q_poseList)

                    # if right_qpos is not None and left_qpos is not None:
                    #     # 4,5: index 6,7: middle, 0,1: pinky, 2,3: ring, 8,9: thumb
                    #     right_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
                    #     right_angles.append(1.2 - right_qpos[8])
                    #     right_angles.append(0.5 - right_qpos[9])
                    #
                    #     left_angles = [1.7- left_qpos[i] for i in  [4, 6, 2, 0]]
                    #     left_angles.append(1.2 - left_qpos[8])
                    #     left_angles.append(0.5 - left_qpos[9])
                    #     h1hand.crtl(right_angles,left_angles)
                    # time.sleep(0.2)

                    # Control all joints including fingers
                    for i in range(model.nu):  # model.nu is the number of actuators
                        actuator_name = data.actuator(i).name
                    
                        try:
                            correspond_joint = data.joint(actuator_name)
                        except KeyError:
                            continue
                        correspond_joint_id = correspond_joint.id

                        print("i, Actuator name:", i, actuator_name)
                        print("Correspond joint id:", correspond_joint_id)
                        # print(len(data.qpos))
                        
                        # Handle arm joints (11-18)
                        if i >= 11 and i <= 18:
                            # data.ctrl[i] = (
                            #     0  # feedforward term tau
                            #     + 800 * (sol_q_values[i - 11] - data.qpos[correspond_joint_id])  # position control kp
                            #     + 20 * (tau_ff_values[i - 11] - data.qvel[correspond_joint_id])  # velocity control kd
                            # )
                            data.qpos[correspond_joint_id] = sol_q_values[i - 11]


                            # print("Control arm:", actuator_name, data.ctrl[i])
                        # Handle left hand fingers (19-30)
                        elif i >= 19 and i <= 42:  # Handle both left (19-30) and right (31-42) hand fingers
                            target_angle = None
                            
                            if i <= 30:  # Left hand
                                angles = left_angles
                                prefix = "l_"
                            else:  # Right hand
                                angles = right_angles
                                prefix = "r_"
                                
                            if f"{prefix}thumb" in actuator_name:
                                if "proximal_1" in actuator_name:
                                    target_angle = angles[4]  # thumb yaw
                                elif "proximal_2" in actuator_name:
                                    target_angle = angles[5]  # thumb pitch
                            elif "proximal" in actuator_name:
                                if f"{prefix}index" in actuator_name:
                                    target_angle = angles[0]
                                elif f"{prefix}middle" in actuator_name:
                                    target_angle = angles[1]
                                elif f"{prefix}ring" in actuator_name:
                                    target_angle = angles[2]
                                elif f"{prefix}pinky" in actuator_name:
                                    target_angle = angles[3]

                            
                            # breakpoint()
                             
                            if target_angle is not None:
                                # print("q_pos:", data.qpos[correspond_joint_id])
                                # print("target_angle:", target_angle)
                                # data.ctrl[i] = (
                                #     0  # feedforward term tau
                                #     + 800 * (target_angle - data.qpos[correspond_joint_id])  # position control kp
                                #     + 20 * (0 - data.qvel[correspond_joint_id])  # velocity control kd
                                # )
                                data.qpos[correspond_joint_id] = target_angle

                                # print("Control finger:", actuator_name, data.ctrl[i])


                    mujoco.mj_step(model, data)
                    renderer.update_scene(data, camera=cam)

                    left_eye_image = renderer.render()
                
                    # Render right eye image
                    # Adjust camera azimuth slightly to the right for stereo effect
                    # cam.azimuth = original_azimuth + azimuth_diff  # Shift camera view right by a small angle
                    right_eye_image = renderer.render()
                    
                    # Restore original camera azimuth
                    # cam.azimuth = original_azimuth
                    
                    # Combine left and right eye images horizontally
                    rendered_image = np.hstack((left_eye_image, right_eye_image))


                    # --- Display using GLFW + OpenGL --- 
                    if rendered_image.shape == teleoperator.img_array.shape:
                        np.copyto(teleoperator.img_array, rendered_image)
                        # sm.write_image(vr_image) # Preserving related commented call

                        # ---------------------------------------------------

                        # --- Explicitly set context and draw to GLFW window ---
                        if window: # Check if window exists
                            try:
                                # Attempt to make the display window's context current
                                glfw.make_context_current(window)

                                # Upload image to texture
                                glBindTexture(GL_TEXTURE_2D, texture_id)
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0,
                                            GL_RGB, GL_UNSIGNED_BYTE, rendered_image)

                                # Prepare to draw
                                glViewport(0, 0, window_width, window_height)
                                glClear(GL_COLOR_BUFFER_BIT)
                                glMatrixMode(GL_PROJECTION)
                                glLoadIdentity()
                                glOrtho(0, window_width, 0, window_height, -1, 1)
                                glMatrixMode(GL_MODELVIEW)
                                glLoadIdentity()

                                # Enable texturing and bind our texture
                                glEnable(GL_TEXTURE_2D)
                                glBindTexture(GL_TEXTURE_2D, texture_id)

                                # Draw a textured quad covering the window
                                glBegin(GL_QUADS)
                                glTexCoord2f(0, 1); glVertex2f(0, 0)
                                glTexCoord2f(1, 1); glVertex2f(window_width, 0)
                                glTexCoord2f(1, 0); glVertex2f(window_width, window_height)
                                glTexCoord2f(0, 0); glVertex2f(0, window_height)
                                glEnd()

                                # Disable texturing and unbind
                                glDisable(GL_TEXTURE_2D)
                                glBindTexture(GL_TEXTURE_2D, 0)

                                # Swap front and back buffers
                                glfw.swap_buffers(window)

                            except Exception as e:
                                # Catch errors during context switching or drawing
                                print(f"ERROR during GLFW display update: {e}")
                                # Decide how to handle: break, continue, log, etc.
                                print("Breaking simulation loop due to display error.")
                                break 
                        # --------------------------------------------------------

                    else:
                        print(f"ERROR: Rendered image shape {rendered_image.shape} does not match shared memory shape {teleoperator.img_array.shape}")
                        break # Stop if shapes don't match

                    # Poll for and process events for the GLFW window
                    glfw.poll_events()


                    # Visualize at fixed FPS.
                    viewer.sync()

        except KeyboardInterrupt:
            print("Shutting down...")
            exit(0)
