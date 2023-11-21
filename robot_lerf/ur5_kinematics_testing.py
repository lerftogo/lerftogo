from ur5.ur5_kinematics import UR5Kinematics
import numpy as np
from autolab_core import RigidTransform
from tracikpy import TracIKSolver
from ur5_robot import UR5RobotKinematics
from cmath import inf









if __name__ == '__main__':
    # urdf_path="/home/lawrence/dmodo/ur5_go_old/ur5_pybullet/urdf/real_arm_w_camera.urdf"
    urdf_path = "/home/lawrence/robotlerf/ur5bc/ur5/ur5.urdf"


    """
    Joint angles:
    [-3.31606681, -1.78024561,  2.46092653, -2.26895982, -1.57074386, 2.93207955]
    Ground truth target pose:
    array([[-0.03704775,  0.99917011,  0.01692807,  0.32928631],
       [ 0.99929832,  0.03713522, -0.00488237,  0.05314915],
       [-0.00550695,  0.01673531, -0.99984479,  0.00744493],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    Joint angles:
    [-3.66179473, -1.19205934,  2.15678024, -2.51019794, -1.5436228, 2.57774162]
    array([[-0.0466848 ,  0.99886794, -0.00913055,  0.46460387],
       [ 0.99827923,  0.04697809,  0.03509471, -0.13132936],
       [ 0.03548391, -0.00747645, -0.99934228, -0.09273899],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    """



    print("====================method 1====================")
    kinematics = UR5Kinematics(urdf_path=urdf_path)
    kinematics.set_tcp(RigidTransform(translation=[0,0.0,.17], from_frame="tcp", to_frame="ee_link"))
    # import pdb;pdb.set_trace()
    # q = np.array([-179/180*np.pi, -90/180*np.pi, 127/180*np.pi, -125/180*np.pi, -91/180*np.pi, 184/180*np.pi])
    # q = np.array([-3.31606681, -1.78024561,  2.46092653, -2.26895982, -1.57074386, 2.93207955])
    q = np.array([-3.66179473, -1.19205934,  2.15678024, -2.51019794, -1.5436228, 2.57774162])
    print("Ground truth:", q / np.pi * 180)
    # target_pose = np.array([[-0.03704775,  0.99917011,  0.01692807,  0.32928631],
    #    [ 0.99929832,  0.03713522, -0.00488237,  0.05314915],
    #    [-0.00550695,  0.01673531, -0.99984479,  0.00744493],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    target_pose = np.array([[-0.0466848 ,  0.99886794, -0.00913055,  0.46460387],
       [ 0.99827923,  0.04697809,  0.03509471, -0.13132936],
       [ 0.03548391, -0.00747645, -0.99934228, -0.09273899],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    

    pose = kinematics.fk(q)
    print("pose",pose)
    solved_q = kinematics.ik(pose, q)
    print("solved_q",solved_q / np.pi * 180)




    print("====================method 2====================")
    ur5 = UR5RobotKinematics(ee_offset='gripper_tip')
    ur5.set_config(q)
    print(ur5.ee_frame)
    print(ur5.ika8(target_pose))
    ur5.ik(target_pose)
    print(ur5.config / np.pi * 180)
    # ur5_collision = CollisionChecker(n_proc=1)
        # link_poses = ur5_collision.fk(np.array(initial_config)[None,...])
    # ee_pose = link_poses["wrist_3_link"]

    print("====================method 3====================")
    ik_solver = TracIKSolver(
        urdf_path,
        "base_link",
        "wrist_3_link",
    )
    sol = ik_solver.ik(target_pose,qinit = q,bx=1e-3,
            by=1e-3,
            bz=1e-3,
            brx=inf,
            bry=inf,
            brz=inf,)
    print(sol / np.pi * 180)