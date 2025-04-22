import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from pathlib import Path

class H1FingerTest:
    def __init__(self):
        # Load the scene XML file
        _HERE = Path(__file__).parent
        xml_path = _HERE / "unitree_h1" / "empty.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
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
    
    def print_joint_info(self):
        """Print information about joints and their qpos indices"""
        print("\nJoint Information:")
        print("-" * 50)
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_idx = self.model.jnt_qposadr[i]
            range_min = self.model.jnt_range[i, 0]
            range_max = self.model.jnt_range[i, 1]
            print(f"Joint {i}: {name} -> qpos[{qpos_idx}]: range [{range_min:.2f}, {range_max:.2f}]")
        print("-" * 50)
    
    def test_finger_movement(self, hand='left'):
        """Test finger movements for specified hand"""
        print(f"\nTesting {hand} hand finger movements...")
        
        # Define joint ranges for each finger
        finger_ranges = {
            'thumb': {
                'proximal_1': (0, 1.3),  # yaw
                'proximal_2': (0, 0.68),  # pitch
                'middle': (0, 0.68),
                'distal': (0, 0.77)
            },
            'index': {
                'proximal': (0, 1.62),
                'distal': (0, 1.82)
            },
            'middle': {
                'proximal': (0, 1.62),
                'distal': (0, 1.82)
            },
            'ring': {
                'proximal': (0, 1.62),
                'distal': (0, 1.75)
            },
            'pinky': {
                'proximal': (0, 1.62),
                'distal': (0, 1.87)
            }
        }
        
        # Test each finger one by one
        for finger, ranges in finger_ranges.items():
            print(f"\nTesting {finger} finger...")
            
            for joint, (min_val, max_val) in ranges.items():
                print(f"Moving {joint} joint...")
                
                # Find the corresponding joint
                for i in range(self.model.njnt):
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name and f"{hand}_{finger}" in joint_name and joint in joint_name:
                        print(f"Found joint: {joint_name}")
                        qpos_idx = self.model.jnt_qposadr[i]
                        
                        # Move joint through its range
                        for t in range(100):
                            # Calculate position (sinusoidal movement)
                            pos = min_val + (max_val - min_val) * (1 + np.sin(t * 0.1)) / 2
                            self.data.qpos[qpos_idx] = pos
                            
                            # Step simulation
                            mujoco.mj_step(self.model, self.data)
                            self.viewer.sync()
                            time.sleep(0.01)
                        
                        # Reset position for this joint
                        self.data.qpos[qpos_idx] = 0
                        break
    
    def test_grasping_pose(self, hand='left'):
        """Test a grasping pose for specified hand"""
        print(f"\nTesting {hand} hand grasping pose...")
        
        # Define grasping positions for each finger
        grasping_positions = {
            'thumb': {
                'proximal_1': 0.8,  # yaw
                'proximal_2': 0.4,  # pitch
                'middle': 0.4,
                'distal': 0.4
            },
            'index': {
                'proximal': 1.0,
                'distal': 1.2
            },
            'middle': {
                'proximal': 1.0,
                'distal': 1.2
            },
            'ring': {
                'proximal': 1.0,
                'distal': 1.2
            },
            'pinky': {
                'proximal': 1.0,
                'distal': 1.2
            }
        }
        
        # # Test shoulder movement
        # print("\nTesting shoulder movement...")
        # shoulder_joints = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw']
        # for joint in shoulder_joints:
        #     print(f"\nTesting {joint}...")
        #     for i in range(self.model.njnt):
        #         joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        #         if joint_name and joint in joint_name:
        #             qpos_idx = self.model.jnt_qposadr[i]
                    
        #             # Move joint through range [-0.5, 0.5]
        #             for t in range(100):
        #                 pos = -0.5 + np.sin(t * 0.1)  # Sinusoidal movement
        #                 self.data.qpos[qpos_idx] = pos
        #                 mujoco.mj_step(self.model, self.data)
        #                 self.viewer.sync()
        #                 time.sleep(0.01)
                    
        #             # Reset position
        #             self.data.qpos[qpos_idx] = 0
        #             break

        # Set all fingers to dynamic sinusoidal movement
        for finger, positions in grasping_positions.items():
            for joint, pos in positions.items():
                # Find the corresponding joint
                for i in range(self.model.njnt):
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    print(f"Testing {joint_name}")
                    if joint_name and f"{hand[0]}_{finger}" in joint_name and joint in joint_name:
                        qpos_idx = self.model.jnt_qposadr[i]
                        # Move joint through sinusoidal pattern for 100 steps
                        for t in range(100):
                            # Scale the movement to be centered around the target position
                            dynamic_pos = pos + 10 * np.sin(t * 0.1)  # Amplitude of 0.3
                            self.data.ctrl[qpos_idx] = dynamic_pos
                            mujoco.mj_step(self.model, self.data)
                            self.viewer.sync()
                            time.sleep(0.01)
                            print(f"Moving {joint_name} to {dynamic_pos:.2f}")
                        break
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        
        # Hold the pose for 5 seconds
        time.sleep(5)
    
    def run(self):
        """Run the finger test sequence"""
        try:
            print("Starting finger movement test...")
            print("Press Ctrl+C to exit")
            
            # Print joint information
            self.print_joint_info()
            
            # Test left hand
            self.test_finger_movement(hand='left')
            time.sleep(1)
            
            # Test right hand
            self.test_finger_movement(hand='right')
            time.sleep(1)
            
            # Test grasping poses
            self.test_grasping_pose(hand='left')
            self.test_grasping_pose(hand='right')
            
            # Keep the viewer open
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    try:
        test = H1FingerTest()
        test.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
        raise 