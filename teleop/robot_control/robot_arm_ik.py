import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Arm_IK:
    def __init__(self, urdf_path=None, mesh_path=None, hand_angles=None):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # Default paths if none provided
        if urdf_path is None:
            urdf_path = '../assets/h1/h1_with_hand.urdf'
        if mesh_path is None:
            mesh_path = '../assets/h1/'
            
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_path)

        # Print all available joint names in the model
        print("\n=== Available Joints in Robot Model ===")
        for i, joint in enumerate(self.robot.model.joints):
            joint_name = self.robot.model.names[i]
            joint_id = joint.id
            print(f"Joint {i}: {joint_name}, ID: {joint_id}")
        print("=======================================\n")

        # Store finger angles for visualization
        self.hand_angles = {} if hand_angles is None else hand_angles
        
        self.mixed_jointsToLockIDs = ["torso_joint"]
        # Note: We're not locking hand joints anymore

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        print("\n=== Available Joints in Reduced Robot Model ===")
        for i, joint in enumerate(self.reduced_robot.model.joints):
            joint_name = self.reduced_robot.model.names[i]
            print(f"Joint {i}: {joint_name}, ID: {joint.id}")
        print("=======================================\n")

        # Add frames for hands and arms
        self.L_ee_id = self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_elbow_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.2605 + 0.05, 0, 0]).T),
                      pin.FrameType.OP_FRAME)
        )

        self.R_ee_id = self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_elbow_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.2605 + 0.05, 0, 0]).T),
                      pin.FrameType.OP_FRAME)
        )

        # Add hand palm frames if available
        self.L_palm_id = None
        self.R_palm_id = None
        
        # Try different possible joint names for the hands
        left_hand_joint_names = ['left_hand_joint']
        right_hand_joint_names = ['right_hand_joint']
        
        for joint_name in left_hand_joint_names:
            try:
                joint_id = self.reduced_robot.model.getJointId(joint_name)
                if joint_id != -1:
                    self.L_palm_id = self.reduced_robot.model.addFrame(
                        pin.Frame('L_palm',
                                joint_id,
                                pin.SE3(np.eye(3), np.zeros(3)),
                                pin.FrameType.OP_FRAME)
                    )
                    print(f"Added left palm frame using joint {joint_name}")
                    break
            except Exception as e:
                continue
        
        if self.L_palm_id is None:
            print("Could not add left hand frame: No valid joint found")
            
        for joint_name in right_hand_joint_names:
            try:
                joint_id = self.reduced_robot.model.getJointId(joint_name)
                if joint_id != -1:
                    self.R_palm_id = self.reduced_robot.model.addFrame(
                        pin.Frame('R_palm',
                                joint_id,
                                pin.SE3(np.eye(3), np.zeros(3)),
                                pin.FrameType.OP_FRAME)
                    )
                    print(f"Added right palm frame using joint {joint_name}")
                    break
            except Exception as e:
                continue
                
        if self.R_palm_id is None:
            print("Could not add right hand frame: No valid joint found")

        # Add finger frames - use automatic detection
        self.finger_frames = {}
        
        # Identify valid joint names that contain finger-related strings
        valid_finger_joints = []
        for i, joint in enumerate(self.reduced_robot.model.joints):
            joint_name = self.reduced_robot.model.names[i]
            # Look for finger-related keywords in joint names
            if any(keyword in joint_name.lower() for keyword in 
                  ["thumb", "index", "middle", "ring", "pinky", "finger"]):
                valid_finger_joints.append(joint_name)
                print(f"Found finger joint: {joint_name}")
                
                # Try to add a frame for this finger joint
                try:
                    frame_id = self.reduced_robot.model.addFrame(
                        pin.Frame(f'{joint_name}_frame',
                                joint.id,
                                pin.SE3(np.eye(3), np.zeros(3)),
                                pin.FrameType.OP_FRAME)
                    )
                    self.finger_frames[joint_name] = frame_id
                    print(f"Added frame for {joint_name}, frame ID: {frame_id}")
                except Exception as e:
                    print(f"Could not add frame for {joint_name}: {e}")

        # Build mapping from generic finger names to actual joint names for control
        self.finger_mapping = {
            # Left hand
            "l_thumb_yaw": None,
            "l_thumb_pitch": None,
            "l_index": None,
            "l_middle": None,
            "l_ring": None,
            "l_pinky": None,
            # Right hand
            "r_thumb_yaw": None,
            "r_thumb_pitch": None,
            "r_index": None,
            "r_middle": None,
            "r_ring": None,
            "r_pinky": None
        }
        
        # Map generic names to actual joint names
        for joint_name in valid_finger_joints:
            # Left hand joints
            if joint_name.startswith('L_'):
                if joint_name == "L_thumb_proximal_yaw_joint":
                    self.finger_mapping["l_thumb_yaw"] = joint_name
                elif joint_name == "L_thumb_proximal_pitch_joint":
                    self.finger_mapping["l_thumb_pitch"] = joint_name
                elif joint_name == "L_index_proximal_joint":
                    self.finger_mapping["l_index"] = joint_name
                elif joint_name == "L_middle_proximal_joint":
                    self.finger_mapping["l_middle"] = joint_name
                elif joint_name == "L_ring_proximal_joint":
                    self.finger_mapping["l_ring"] = joint_name
                elif joint_name == "L_pinky_proximal_joint":
                    self.finger_mapping["l_pinky"] = joint_name
            
            # Right hand joints
            if joint_name.startswith('R_'):
                if joint_name == "R_thumb_proximal_yaw_joint":
                    self.finger_mapping["r_thumb_yaw"] = joint_name
                elif joint_name == "R_thumb_proximal_pitch_joint":
                    self.finger_mapping["r_thumb_pitch"] = joint_name
                elif joint_name == "R_index_proximal_joint":
                    self.finger_mapping["r_index"] = joint_name
                elif joint_name == "R_middle_proximal_joint":
                    self.finger_mapping["r_middle"] = joint_name
                elif joint_name == "R_ring_proximal_joint":
                    self.finger_mapping["r_ring"] = joint_name
                elif joint_name == "R_pinky_proximal_joint":
                    self.finger_mapping["r_pinky"] = joint_name

        print("\n=== Finger Joint Mapping ===")
        for generic_name, actual_name in self.finger_mapping.items():
            print(f"{generic_name}: {actual_name}")
        print("===========================\n")

        self.init_data = np.zeros(self.reduced_robot.model.nq)

        print(f"Initial data length: {len(self.init_data)}")
        
        # Print joint mapping for all elements in self.init_data
        self.print_init_data_mapping()

        # Collect frame IDs for visualization
        display_frames = [self.L_ee_id, self.R_ee_id]
        if self.L_palm_id:
            display_frames.append(self.L_palm_id)
        if self.R_palm_id:
            display_frames.append(self.R_palm_id)
            
        display_frames.extend(self.finger_frames.values())

        # Initialize the Meshcat visualizer with finger frames
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(True, frame_ids=display_frames, axis_length=0.05, axis_width=3)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # Add target frames for hands in visualization
        self._add_target_frame('L_ee_target', [1, 0, 0])  # Red for left arm
        self._add_target_frame('R_ee_target', [0, 0, 1])  # Blue for right arm
        if self.L_palm_id:
            self._add_target_frame('L_palm_target', [0, 1, 0])  # Green for left palm
        if self.R_palm_id:
            self._add_target_frame('R_palm_target', [0, 0.5, 0.5])  # Teal for right palm

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the joint IDs and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        print(f"L_hand_id: {self.L_hand_id}")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        print(f"R_hand_id: {self.R_hand_id}")
        
        # Define the error function
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3]
                )
            ],
        )
        
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        
        # Define costs
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Set constraints
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        
        # Set objective
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False
        }
        self.opti.solver("ipopt", opts)

    def _add_target_frame(self, name, color=[1, 0, 0], axis_length=0.1, axis_width=5):
        """Add a target frame to the visualization with custom color"""
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        
        # Create colors with the base color for each axis
        r, g, b = color
        FRAME_AXIS_COLORS = (
            np.array([[r, 0, 0], [min(r+0.3, 1), 0, 0],
                      [0, g, 0], [0, min(g+0.3, 1), 0],
                      [0, 0, b], [0, 0, min(b+0.3, 1)]]).astype(np.float32).T
        )
        
        self.vis.viewer[name].set_object(
            mg.LineSegments(
                mg.PointsGeometry(
                    position=axis_length * FRAME_AXIS_POSITIONS,
                    color=FRAME_AXIS_COLORS,
                ),
                mg.LineBasicMaterial(
                    linewidth=axis_width,
                    vertexColors=True,
                ),
            )
        )

    def update_finger_visualization(self):
        """Update the visualization of finger positions based on joint angles"""
        # This method will update the visualization for all finger joints
        if self.hand_angles:
            for joint_name, angle in self.hand_angles.items():
                if joint_name in self.finger_frames:
                    # Update the frame position visually
                    frame_id = self.finger_frames[joint_name]
                    joint_id = self.reduced_robot.model.getJointId(joint_name)
                    
                    # Compute forward kinematics for the updated joint positions
                    q = self.init_data.copy()
                    q[joint_id] = angle
                    
                    # This will be called from the main visualization update

    def set_hand_angles(self, left_angles=None, right_angles=None):
        """Set the hand angles for visualization and control based on the detected joint mapping"""
        # Store all available joint names for quick lookup
        available_joints = list(self.finger_frames.keys())
        
        if left_angles is not None:
            # Clear previous hand angles
            new_hand_angles = {}
            
            # [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
            # Update only if we have valid joints mapped
            if self.finger_mapping["l_index"]:
                # Main joint
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["l_index"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["l_index"]] = left_angles[0]
                    # Get corresponding intermediate joint name
                    inter_name = self.finger_mapping["l_index"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = left_angles[0]
            
            if self.finger_mapping["l_middle"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["l_middle"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["l_middle"]] = left_angles[1]
                    inter_name = self.finger_mapping["l_middle"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = left_angles[1]
            
            if self.finger_mapping["l_ring"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["l_ring"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["l_ring"]] = left_angles[2]
                    inter_name = self.finger_mapping["l_ring"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = left_angles[2]
            
            if self.finger_mapping["l_pinky"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["l_pinky"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["l_pinky"]] = left_angles[3]
                    inter_name = self.finger_mapping["l_pinky"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = left_angles[3]
            
            if self.finger_mapping["l_thumb_yaw"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["l_thumb_yaw"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["l_thumb_yaw"]] = left_angles[4]
            
            if self.finger_mapping["l_thumb_pitch"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["l_thumb_pitch"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["l_thumb_pitch"]] = left_angles[5]
                    # Try to find intermediate and distal joints for thumb
                    inter_name = "L_thumb_intermediate_joint"
                    distal_name = "L_thumb_distal_joint"
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = left_angles[5]
                    if distal_name in available_joints:
                        distal_joint_id = self.reduced_robot.model.getJointId(distal_name)
                        if distal_joint_id != -1:
                            distal_config_idx = self.reduced_robot.model.joints[distal_joint_id].idx_q
                            new_hand_angles[distal_name] = left_angles[5]
            
            # Update the hand angles
            self.hand_angles.update(new_hand_angles)
            print(f"Updated left hand angles: {len(new_hand_angles)} joints")
            
        if right_angles is not None:
            # Clear previous hand angles
            new_hand_angles = {}
            
            # [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
            if self.finger_mapping["r_index"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["r_index"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["r_index"]] = right_angles[0]
                    inter_name = self.finger_mapping["r_index"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = right_angles[0]
            
            if self.finger_mapping["r_middle"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["r_middle"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["r_middle"]] = right_angles[1]
                    inter_name = self.finger_mapping["r_middle"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = right_angles[1]
            
            if self.finger_mapping["r_ring"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["r_ring"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["r_ring"]] = right_angles[2]
                    inter_name = self.finger_mapping["r_ring"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = right_angles[2]
            
            if self.finger_mapping["r_pinky"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["r_pinky"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["r_pinky"]] = right_angles[3]
                    inter_name = self.finger_mapping["r_pinky"].replace("proximal", "intermediate")
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = right_angles[3]
            
            if self.finger_mapping["r_thumb_yaw"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["r_thumb_yaw"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["r_thumb_yaw"]] = right_angles[4]
            
            if self.finger_mapping["r_thumb_pitch"]:
                joint_id = self.reduced_robot.model.getJointId(self.finger_mapping["r_thumb_pitch"])
                if joint_id != -1:
                    config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                    new_hand_angles[self.finger_mapping["r_thumb_pitch"]] = right_angles[5]
                    # Try to find intermediate and distal joints for thumb
                    inter_name = "R_thumb_intermediate_joint"
                    distal_name = "R_thumb_distal_joint"
                    if inter_name in available_joints:
                        inter_joint_id = self.reduced_robot.model.getJointId(inter_name)
                        if inter_joint_id != -1:
                            inter_config_idx = self.reduced_robot.model.joints[inter_joint_id].idx_q
                            new_hand_angles[inter_name] = right_angles[5]
                    if distal_name in available_joints:
                        distal_joint_id = self.reduced_robot.model.getJointId(distal_name)
                        if distal_joint_id != -1:
                            distal_config_idx = self.reduced_robot.model.joints[distal_joint_id].idx_q
                            new_hand_angles[distal_name] = right_angles[5]
            
            # Update the hand angles
            self.hand_angles.update(new_hand_angles)
            print(f"Updated right hand angles: {len(new_hand_angles)} joints")

    def ik_fun(self, left_pose, right_pose, motorstate=None, motorV=None, left_hand_angles=None, right_hand_angles=None):
        """Run IK with updated parameters for hand visualization"""
        # Update hand angles if provided
        if left_hand_angles is not None or right_hand_angles is not None:
            self.set_hand_angles(left_hand_angles, right_hand_angles)
            
        if motorstate is not None:
            self.init_data = motorstate
        
        # Build a config index to joint map for debugging
        config_idx_to_joint = {}
        for i, joint in enumerate(self.reduced_robot.model.joints):
            if i > 0:  # Skip the universe joint
                joint_name = self.reduced_robot.model.names[i]
                config_idx = joint.idx_q
                config_idx_to_joint[config_idx] = joint_name
        
        # Apply hand angles to the initial state
        if self.hand_angles:
            for joint_name, angle in self.hand_angles.items():
                try:
                    joint_id = self.reduced_robot.model.getJointId(joint_name)
                    if joint_id != -1:
                        config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                        if config_idx < len(self.init_data):
                            self.init_data[config_idx] = angle
                            # print(f"Setting {joint_name} at index {config_idx} to {angle}")
                        else:
                            print(f"Warning: Config index {config_idx} for joint {joint_name} is out of bounds")
                except Exception as e:
                    print(f"Error setting angle for {joint_name}: {e}")
                    
        self.opti.set_initial(self.var_q, self.init_data)

        left_pose, right_pose = self.adjust_pose(left_pose, right_pose)

        # Update visualization for end effectors and hands
        self.vis.viewer['L_ee_target'].set_transform(left_pose)
        self.vis.viewer['R_ee_target'].set_transform(right_pose)
        
        # Calculate hand poses with slight offset for visualization
        left_palm_pose = left_pose.copy()
        right_palm_pose = right_pose.copy()
        left_palm_pose[:3, 3] += np.array([0.05, 0, 0])  # Offset for visualization
        right_palm_pose[:3, 3] += np.array([0.05, 0, 0])  # Offset for visualization
        
        self.vis.viewer['L_palm_target'].set_transform(left_palm_pose)
        self.vis.viewer['R_palm_target'].set_transform(right_palm_pose)

        self.opti.set_value(self.param_tf_l, left_pose)
        self.opti.set_value(self.param_tf_r, right_pose)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            # Update hand angles in the solution if needed
            if self.hand_angles:
                for joint_name, angle in self.hand_angles.items():
                    try:
                        joint_id = self.reduced_robot.model.getJointId(joint_name)
                        if joint_id != -1:
                            config_idx = self.reduced_robot.model.joints[joint_id].idx_q
                            if config_idx < len(sol_q):
                                sol_q[config_idx] = angle
                                # print(f"Setting solution {joint_name} at index {config_idx} to {angle}")
                            else:
                                print(f"Warning: Config index {config_idx} for joint {joint_name} in solution is out of bounds")
                    except Exception as e:
                        print(f"Error setting angle in solution for {joint_name}: {e}")

            # Update visualization with new joint positions
            self.vis.display(sol_q)
            self.init_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q-self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            return sol_q, tau_ff, True
        
        except Exception as e:
            print(f"ERROR in convergence: {e}")
            return self.init_data, None, False

    def adjust_pose(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def print_init_data_mapping(self):
        """Print a mapping of each configuration index in self.init_data to the corresponding joint name"""
        print("\n=== Mapping of self.init_data indices to joint names ===")
        
        # Create a mapping of configuration indices to joint names
        config_idx_to_joint = {}
        
        for i, joint in enumerate(self.reduced_robot.model.joints):
            if i > 0:  # Skip the universe joint (i=0)
                joint_name = self.reduced_robot.model.names[i]
                config_idx = joint.idx_q
                config_idx_to_joint[config_idx] = joint_name
        
        # Print each index in self.init_data and its corresponding joint name
        for idx in range(len(self.init_data)):
            joint_name = config_idx_to_joint.get(idx, "No mapping found")
            print(f"self.init_data[{idx}] -> {joint_name}")
            
        print("=======================================\n")
        
if __name__ == "__main__":
    arm_ik = Arm_IK()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, +0.2, 0.2]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, -0.2, 0.2]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':

        for i in range(150):
            angle = rotation_speed * i
            L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
            R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

            L_tf_target.translation += np.array([0.001,  0.001, 0.001])
            R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            arm_ik.ik_fun(L_tf_target.homogeneous, R_tf_target.homogeneous)
            time.sleep(0.02)
