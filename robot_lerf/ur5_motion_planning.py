import numpy as np
from autolab_core import RigidTransform
from robot_lerf.ur5_robot import UR5RobotKinematics
from robot_lerf.ur5_collision_checker import CollisionChecker
from ur5py.ur5 import UR5Robot
from robot_lerf.capture_utils import get_blends

class UR5MotionPlanning:
    def __init__(self):
        self.ur5 = UR5RobotKinematics(ee_offset='gripper_center')
        self.ur_collision = CollisionChecker(n_proc=8)

    def test_collision(self, target_joint_angles, current_joint_angles):
        """
        target_joint_angles: np.array of shape (6,)
        current_joint_angles: np.array of shape (6,)
        """
        # do collision checking
        collide = self.ur_collision.in_collision(target_joint_angles[np.newaxis, :])
        if collide:
            print("target joint is in collision")
            return True
        else:
            # check collision for linear interpolation along the path
            num_steps = 100
            traj = np.linspace(current_joint_angles, target_joint_angles, num_steps)
            collide = self.ur_collision.in_collision(traj)
            if collide:
                print("collision in the middle of the trajectory")
                return True
            return False


    def get_trajectory(self, target_pose:RigidTransform, current_joint_angles, allow_180 = True):
        """
        allow_180: whether allow rotating the gripper 180 degrees to avoid collision 
        (eg. if the camera was supposed to inside, rotate it to the outside)
        """
        self.ur5.set_config(current_joint_angles)
        self.ur5.ik(target_pose,ee_offset='gripper_center')
        target_joint_angles = self.ur5.config

        if target_joint_angles[0] - current_joint_angles[0] > np.pi:
            target_joint_angles[0] -= 2*np.pi
        elif target_joint_angles[0] - current_joint_angles[0] < -np.pi:
            target_joint_angles[0] += 2*np.pi

        collide = self.test_collision(target_joint_angles, current_joint_angles)
        
        if collide and allow_180:
            if target_joint_angles[-1] > 0:
                target_joint_angles[-1] -= np.pi
            else:
                target_joint_angles[-1] += np.pi
            collide = self.test_collision(target_joint_angles, current_joint_angles)
            if not collide:
                print("rotate 180 degrees to avoid collision")
        if collide:
            other_joint_angles = self.ur5.ika8(target_pose,ee_offset='gripper_center') # ndarray (6, 8)
            for i in range(8):
                if other_joint_angles[0, i] - current_joint_angles[0] > np.pi:
                    other_joint_angles[0, i] -= 2*np.pi
                elif other_joint_angles[0, i] - current_joint_angles[0] < -np.pi:
                    other_joint_angles[0, i] += 2*np.pi

                collide = self.test_collision(other_joint_angles[:, i], current_joint_angles)
                if not collide:
                    target_joint_angles = other_joint_angles[:, i]
                    break
        if collide:
            print("no valid trajectory found")
            return None, False
        else:
            traj = np.linspace(current_joint_angles, target_joint_angles, 20)
            return traj, True




if __name__ == '__main__':
    robot = UR5Robot()
    current_q = robot.get_joints()

    translation = np.array([ 0.4773733  , 0.01320116 ,-0.050937])
    rotation = np.array(
        RigidTransform.x_axis_rotation(np.pi)
    )

    target_pose = np.eye(4)
    target_pose[:3, :3] = rotation
    target_pose[:3, 3] = translation

    ur5mp = UR5MotionPlanning()
    traj, succ = ur5mp.get_trajectory(target_pose, current_q)
    if succ:
        print("found path")
        robot.move_joint_path(traj,vels=[.2]*len(traj),accs=[1]*len(traj),blends = [0.01]*len(traj),asyn=True)
    else:
        print("coudlnt plan")