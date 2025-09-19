import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# from ik import gx7_ik_solve, gx7_fk
from ik_solve import gx7_ik_solve, gx7_fk

# Define slider parameter ranges
POSITION_MIN = -0.5
POSITION_MAX = 0.5
ORIENTATION_MIN = -np.pi / 2
ORIENTATION_MAX = np.pi / 2
PSI_MIN = 0
PSI_MAX = np.pi

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane
# planeId = p.loadURDF("plane.urdf")

# Get the path to the URDF file
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = "../urdf/gx7.urdf"

# Load the robot
robotId = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

# Get number of joints and joint info
num_joints = p.getNumJoints(robotId)
joint_indices = []
joint_names = []

# Collect joint information
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    if joint_info[2] == p.JOINT_REVOLUTE:  # Only consider revolute joints
        joint_indices.append(i)
        joint_names.append(joint_info[1].decode("utf-8"))

# Get the end effector link index (Link7)
end_effector_index = 6

# Create sliders for target position, orientation, and psi
# Position sliders (x, y, z)
x_slider = p.addUserDebugParameter("Target X", POSITION_MIN, POSITION_MAX, 0.4)
y_slider = p.addUserDebugParameter("Target Y", POSITION_MIN, POSITION_MAX, 0.0)
z_slider = p.addUserDebugParameter("Target Z", POSITION_MIN, POSITION_MAX, 0.2)

# Orientation sliders (roll, pitch, yaw)
roll_slider = p.addUserDebugParameter("Roll", ORIENTATION_MIN, ORIENTATION_MAX, 0)
pitch_slider = p.addUserDebugParameter("Pitch", ORIENTATION_MIN, ORIENTATION_MAX, 0)
yaw_slider = p.addUserDebugParameter("Yaw", ORIENTATION_MIN, ORIENTATION_MAX, 0.0)

# Psi slider (redundancy parameter)
psi_slider = p.addUserDebugParameter("Psi", PSI_MIN, PSI_MAX, np.pi / 2)


lower_limits = [-120, 0, -120, -120, -120, -91, -180]
upper_limits = [120, 180, 120, 0, 120, 91, 180]

lower_limits = np.array(lower_limits) * np.pi / 180
upper_limits = np.array(upper_limits) * np.pi / 180


while True:
    # Read current slider values
    target_position = [
        p.readUserDebugParameter(x_slider),
        p.readUserDebugParameter(y_slider),
        p.readUserDebugParameter(z_slider),
    ]

    # Read orientation sliders
    roll = p.readUserDebugParameter(roll_slider)
    pitch = p.readUserDebugParameter(pitch_slider) + np.pi
    yaw = p.readUserDebugParameter(yaw_slider)
    target_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    # Read psi slider
    psi = p.readUserDebugParameter(psi_slider)

    # Update end-effector pose
    ee_pose = np.eye(4)
    ee_pose[:3, :3] = np.array(p.getMatrixFromQuaternion(target_orientation)).reshape(
        3, 3
    )
    ee_pose[:3, 3] = np.array(target_position)

    # Solve inverse kinematics with current slider values
    ik_output = gx7_ik_solve(ee_pose[:3, 3], ee_pose[:3, :3], psi=psi)

    ik_output_limit = []
    for i, qs in enumerate(ik_output):

        qs_normal = []
        for q, l, u in zip(qs, lower_limits, upper_limits):
            if q > np.pi:
                q = q - 2 * np.pi
            elif q < -np.pi:
                q = q + 2 * np.pi
            qs_normal.append(q)

            if q < l or q > u:
                break
        if len(qs_normal) == len(lower_limits):
            # If all joint angles are within limits, add to the output list
            ik_output_limit.append(np.array(qs_normal).ravel())

    if len(ik_output_limit):
        print(len(ik_output_limit), "solutions within limits")

        # Get current joint positions
        current_joint_poses = []
        for i, joint_idx in enumerate(joint_indices):
            if i < 7:  # We only need the 7 joint angles
                current_joint_poses.append(p.getJointState(robotId, joint_idx)[0])
        current_joint_poses = np.array(current_joint_poses)

        # Find the solution with minimum movement from current position
        min_distance = float("inf")
        best_index = 0
        for i, solution in enumerate(ik_output_limit):
            # Calculate distance (joint space) between current position and solution
            distance = np.sum(np.abs(solution - current_joint_poses))
            if distance < min_distance:
                min_distance = distance
                best_index = i

        print(
            f"Selected solution {best_index} with minimum movement distance: {min_distance:.4f}"
        )
        joint_poses = ik_output_limit[best_index].ravel()

        pos, ori = gx7_fk(joint_poses)
        print("=== FK ====")
        print(f"pos:\n{pos.ravel()}\n ori:\n{ori}")

        # print(f'qs: {[f"{q:.3f}" for q in joint_poses]}')
        # Apply the calculated joint positions to the robot
        for i, joint_idx in enumerate(joint_indices):
            if i < len(joint_poses):
                p.setJointMotorControl2(
                    bodyIndex=robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500,
                )

        ee_link_state = p.getLinkState(robotId, end_effector_index)
        ee_pos = ee_link_state[0]
        ee_ori = ee_link_state[1]

        ee_ori_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
        print("=== PB ====")
        print(f"ee_pos:\n {ee_pos}\n ee_ori:\n {ee_ori_matrix}")

        print(f"diff xyz: {ee_pos - np.array(target_position)}")

    p.stepSimulation()
    time.sleep(0.01)
