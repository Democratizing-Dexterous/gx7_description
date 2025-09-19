import numpy as np
from SEW_IK.IK_3R_R_3R import IK_3R_R_3R
from IK_helpers.subproblem import rot, wrapToPi
from IK_helpers.sew_stereo import sew_stereo
from IK_helpers.fwdkin_inter import fwdkin_inter
import numpy as np

ori_align = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


def gx7_ik_solve(pos, ori, psi=np.pi / 6):

    ori = ori @ ori_align.T

    kin = {}
    kin["joint_type"] = [0, 0, 0, 0, 0, 0, 0]
    kin["H"] = np.array(
        [[0, 0, -1, 0, 1, 0, 1], [0, 1, 0, -1, 0, -1, 0], [1, 0, 0, 0, 0, 0, 0]]
    )
    kin["P"] = np.array(
        [
            [0, 0, 0, -0.32099, 0.2670, 0, 0, 0],
            [0, 0, 0, -0.00000001, 0, 0, 0, 0],
            [0.1195, 0, 0, 0.077, 0, 0, 0, 0],
        ]
    )

    pos = pos.reshape(3, 1)

    SEW = sew_stereo(np.array([[0], [0], [-1]]), np.array([[0], [1], [0]]))
    Q, is_LS_vec = IK_3R_R_3R(ori, pos, SEW, psi, kin)
    return Q.T


def gx7_fk(joint_angles):
    kin = {}
    kin["joint_type"] = [0, 0, 0, 0, 0, 0, 0]

    kin["H"] = np.array(
        [[0, 0, -1, 0, 1, 0, 1], [0, 1, 0, -1, 0, -1, 0], [1, 0, 0, 0, 0, 0, 0]]
    )
    kin["P"] = np.array(
        [
            [0, 0, 0, -0.32099, 0.2670, 0, 0, 0],
            [0, 0, 0, -0.00000001, 0, 0, 0, 0],
            [0.1195, 0, 0, 0.077, 0, 0, 0, 0],
        ]
    )

    joint_angles = joint_angles.reshape(-1)
    ori, pos, swe = fwdkin_inter(kin, joint_angles, inter=[1, 3, 5])

    return pos, ori @ ori_align


if __name__ == "__main__":
    # Test the IK solver with a sample position and orientation
    target_position = np.array([0, 0, 0.42])
    target_orientation = np.eye(3)  # Identity matrix for no rotation

    ik_solutions = gx7_ik_solve(target_position, target_orientation, psi=np.pi / 6)

    print("IK Solutions:")
    for sol in ik_solutions:
        print(sol)
        pos, ori = gx7_fk(sol)
        print(f"FK Position: {pos}, FK Orientation: \n{ori}\n")
