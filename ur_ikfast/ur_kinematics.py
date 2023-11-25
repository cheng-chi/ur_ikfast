import numpy as np
from scipy.spatial.transform import Rotation

class URKinematics():
    """
    The base frame used for ikfast is off by a 180 degree rotation 
    around the base with the real robot, as shown in:
    https://github.com/cheng-chi/ur_ikfast/blob/3c9c8b5a1400ea7946b609a4097adae9c8d3c7a8/ur5e/ur5e.urdf#L319
    Therefore, we compensate this rotation to make outout consistent with real robot.
    """
    rx_ikbase_realbase = np.array([
        [-1,0,0],
        [0,-1,0],
        [0,0,1]
    ], dtype=np.float64)

    def __init__(self, robot_name):
        if robot_name == 'ur3':
            import ur3_ikfast as ur_ikfast
        elif robot_name == 'ur3e':
            import ur3e_ikfast as ur_ikfast
        elif robot_name == 'ur5':
            import ur5_ikfast as ur_ikfast
        elif robot_name == 'ur5e':
            import ur5e_ikfast as ur_ikfast
        elif robot_name == 'ur10':
            import ur10_ikfast as ur_ikfast
        elif robot_name == 'ur10e':
            import ur10e_ikfast as ur_ikfast
        else:
            raise Exception("Unsupported robot {}".format(robot_name))

        self.kinematics = ur_ikfast.PyKinematics()
        self.n_joints = self.kinematics.getDOF()

    def forward(self, joint_angles, rotation_type='axis_angle'):
        """
            Compute robot's forward kinematics for the specified robot
            joint_angles: list
            rotation_type: 'axis_angle' or 'quaternion' or 'matrix'
            :return: if 'quaternion' then return a list of [x, y, z, w. qx, qy, qz]
                     if 'matrix' then a list of 12 values the 3x3 rotation matrix and 
                     the 3 translational values
        """
        if isinstance(joint_angles, np.ndarray):
            joint_angles = joint_angles.tolist()

        # convert pose to be consistent with real robot
        tx_ikbase_tcp = self.kinematics.forward(joint_angles)
        tx_ikbase_tcp = np.asarray(tx_ikbase_tcp).reshape(3, 4)
        # rotation transpose equals inverse
        tx_realbase_tcp = self.rx_ikbase_realbase.T @ tx_ikbase_tcp

        if rotation_type == 'matrix':
            return tx_realbase_tcp
        elif rotation_type == 'quaternion':
            xyzw = Rotation.from_matrix(tx_realbase_tcp[:,:3]).as_quat()
            pose = np.concatenate([tx_realbase_tcp[:,3], xyzw[[3,0,1,2]]])
            return pose
        elif rotation_type == 'axis_angle':
            rotvec = Rotation.from_matrix(tx_realbase_tcp[:,:3]).as_rotvec()
            pose = np.concatenate([tx_realbase_tcp[:,3], rotvec])
            return pose
        else:
            raise RuntimeError("Unsupported rotation_type: {}".format(rotation_type))

    def inverse(self, ee_pose, all_solutions=False, q_guess=np.zeros(6)):
        """ Compute robot's inverse kinematics for the specified robot
            ee_pose: list of 6 if axis_angle [x, y, z, rx, ry, rz]
                     list of 7 if quaternion [x, y, z, w, qx, qy, qz]
                     list of 12 if rotation matrix + translational values
            all_solutions: whether to return all the solutions found or just the best one
            q_guess:  if just one solution is request, this set of joint values will be use
                      to find the closest solution to this
            :return: list of joint angles
                     list of best joint angles if found
                     q_guess if no solution is found
        """
        ee_pose = np.array(ee_pose)
        
        pose = None
        if ee_pose.shape == (6,):
            pos = ee_pose[:3]
            rot = Rotation.from_rotvec(ee_pose[3:])
            mat = np.zeros((3,4))
            mat[:,:3] = rot.as_matrix()
            mat[:,3] = pos
            pose = mat
        elif ee_pose.shape == (6,):
            pos = ee_pose[:3]
            rot = Rotation.from_quat(ee_pose[3:][[1,2,3,0]])
            mat = np.zeros((3,4))
            mat[:,:3] = rot.as_matrix()
            mat[:,3] = pos
            pose = mat
        elif ee_pose.shape == (4,4):
            pose = ee_pose[:3]
        else:
            pose = ee_pose
        
        # convert pose to be consistent with real robot
        tx_realbase_tcp = pose.reshape(3,4)
        tx_ikbase_tcp = self.rx_ikbase_realbase @ tx_realbase_tcp
        
        joint_configs = self.kinematics.inverse(tx_ikbase_tcp.flatten().tolist())
        n_solutions = int(len(joint_configs)/self.n_joints)
        joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)

        if all_solutions:
            return joint_configs

        return best_ik_sol(joint_configs, q_guess)


def best_ik_sol(sols, q_guess, weights=np.ones(6)):
    """ Get best IK solution """
    valid_sols = []
    for sol in sols:
        test_sol = np.ones(6) * 9999.
        for i in range(6):
            for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2. * np.pi
                        and abs(test_ang - q_guess[i]) <
                        abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if not valid_sols:
        return None
    best_sol_ind = np.argmin(
        np.sum((weights * (valid_sols - np.array(q_guess)))**2, 1))
    return valid_sols[best_sol_ind]
