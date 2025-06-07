import numpy as np
import mujoco
import mujoco.viewer
import time
import os

class H1MujocoManualTest:
    def __init__(self):
        # Load the scene XML file
        xml_path = os.path.join(os.path.dirname(__file__), "./unitree_robots/h1_inspire/scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Calculate and print total mass of the robot
        total_mass = 0.0
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            total_mass += body_mass
        print(f"Total robot mass: {total_mass:.2f} kg")

        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # PD control parameters
        self.kp = 1000.0  # Position gain
        self.kd = 100.0   # Velocity gain
        
        # Print model information
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of bodies: {self.model.nbody}")
        print(f"Number of actuators: {self.model.nu}")
        
        # Print joint names and ranges
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            range_min = self.model.jnt_range[i, 0]
            range_max = self.model.jnt_range[i, 1]
            print(f"Joint {i}: {name}: range [{range_min:.2f}, {range_max:.2f}]")
    
    def print_joint_qpos_correspondences(self):
        """Print correspondences between joint names and qpos indices"""
        print("\nJoint name to qpos index correspondences:")
        print("-" * 50)
        
        # Get all joint names and their qpos indices
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = self.model.jnt_type[i]
            
            # Get qpos index for this joint
            qpos_idx = self.model.jnt_qposadr[i]
            
            # Get joint range
            range_min = self.model.jnt_range[i, 0]
            range_max = self.model.jnt_range[i, 1]
            
            # Get joint type string
            type_str = "hinge" if joint_type == 0 else "slide" if joint_type == 1 else "ball" if joint_type == 2 else "free"
            
            print(f"qpos[{qpos_idx}]: {joint_name} (type: {type_str}, range: [{range_min:.2f}, {range_max:.2f}])")
        
        print("-" * 50)
    
    def pd_control(self, target_pos, joint_idx, dt=0.01):
        """Apply PD control to a joint"""
        # Get current position and velocity
        current_pos = self.data.qpos[joint_idx]
        current_vel = self.data.qvel[joint_idx]
        
        # Calculate position and velocity errors
        pos_error = target_pos - current_pos
        vel_error = 0 - current_vel  # Target velocity is 0
        
        # Calculate control signal
        control = self.kp * pos_error + self.kd * vel_error
        
        # Apply control to the joint
        self.data.ctrl[joint_idx] = - control
        
        # Print debug info
        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
        print(f"Joint {joint_idx} ({joint_name}): target={target_pos:.3f}, current={current_pos:.3f}, control={control:.3f}")
    
    def test_arm_movement(self):
        """Test arm joint movements with PD control"""
        print("\nTesting arm movements...")
        
        # Test left arm joints
        print("\nTesting left arm joints:")
        # 15: left_shoulder_pitch_joint
        # 16: left_shoulder_roll_joint
        # 17: left_shoulder_yaw_joint
        # 18: left_elbow_joint
        
        # Test each joint individually
        for i in range(100):
            # Oscillate left shoulder pitch
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 15)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            # Oscillate left shoulder roll
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 16)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            # Oscillate left shoulder yaw
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 17)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            # Oscillate left elbow
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 18)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        # Test right arm joints
        print("\nTesting right arm joints:")
        # 32: right_shoulder_pitch_joint
        # 33: right_shoulder_roll_joint
        # 34: right_shoulder_yaw_joint
        # 35: right_elbow_joint
        
        for i in range(100):
            # Oscillate right shoulder pitch
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 32)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            # Oscillate right shoulder roll
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 33)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            # Oscillate right shoulder yaw
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 34)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            # Oscillate right elbow
            target_pos = np.sin(i * 0.1) * 0.5
            self.pd_control(target_pos, 35)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
    
    def test_hand_movement(self):
        """Test hand joint movements with PD control"""
        print("\nTesting hand movements...")
        
        # Test left hand joints
        print("\nTesting left hand joints:")
        # 19: left_hand_joint
        # 20: L_thumb_proximal_yaw_joint
        # 21: L_thumb_proximal_pitch_joint
        # 22: L_thumb_intermediate_joint
        # 23: L_thumb_distal_joint
        # 24: L_index_proximal_joint
        # 25: L_index_intermediate_joint
        # 26: L_middle_proximal_joint
        # 27: L_middle_intermediate_joint
        # 28: L_ring_proximal_joint
        # 29: L_ring_intermediate_joint
        # 30: L_pinky_proximal_joint
        # 31: L_pinky_intermediate_joint
        
        # Test thumb joints one by one
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # thumb yaw
            self.pd_control(target_pos, 20)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # thumb pitch
            self.pd_control(target_pos, 21)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        # Test fingers one by one
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # index proximal
            self.pd_control(target_pos, 24)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # middle proximal
            self.pd_control(target_pos, 26)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # ring proximal
            self.pd_control(target_pos, 28)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # pinky proximal
            self.pd_control(target_pos, 30)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        # Test right hand joints
        print("\nTesting right hand joints:")
        # 36: right_hand_joint
        # 37: R_thumb_proximal_yaw_joint
        # 38: R_thumb_proximal_pitch_joint
        # 39: R_thumb_intermediate_joint
        # 40: R_thumb_distal_joint
        # 41: R_index_proximal_joint
        # 42: R_index_intermediate_joint
        # 43: R_middle_proximal_joint
        # 44: R_middle_intermediate_joint
        # 45: R_ring_proximal_joint
        # 46: R_ring_intermediate_joint
        # 47: R_pinky_proximal_joint
        # 48: R_pinky_intermediate_joint
        
        # Test thumb joints one by one
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # thumb yaw
            self.pd_control(target_pos, 37)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # thumb pitch
            self.pd_control(target_pos, 38)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        # Test fingers one by one
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # index proximal
            self.pd_control(target_pos, 41)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # middle proximal
            self.pd_control(target_pos, 43)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # ring proximal
            self.pd_control(target_pos, 45)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
        
        for i in range(100):
            target_pos = np.sin(i * 0.1) * 0.3  # pinky proximal
            self.pd_control(target_pos, 47)
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.01)
    
    def run(self):
        """Run the manual test sequence"""
        try:
            print("Starting manual joint test...")
            print("Press Ctrl+C to exit")
            
            # Print joint-qpos correspondences first
            self.print_joint_qpos_correspondences()
            
            # Test arm movements
            self.test_arm_movement()
            time.sleep(1)
            
            # Test hand movements
            self.test_hand_movement()
            
            # Keep the viewer open
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    try:
        test = H1MujocoManualTest()
        test.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
        raise 