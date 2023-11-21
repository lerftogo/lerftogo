import numpy as np
from numpy import pi
import math
from robot_lerf.transform import rotationXsc, rotationX, rotationYsc, rotationY, rotationZ, rotationZsc, translation
def mod_range(a):
    """Returns a modulo 2pi such that it is within [-pi, pi] instead of [0, 2pi)"""
    return (a-np.pi) % (2*np.pi) - np.pi
# def _Rxsc(s, c):
#     """Create a rotation matrix about the x-axis

#     This version takes a sin and cos value as arguments and populates
#     the appropriate parts of a 4x4 matrix.
#     """
#     return np.asarray([
#         [ 1, 0, 0, 0 ],
#         [ 0, c,-s, 0 ],
#         [ 0, s, c, 0 ],
#         [ 0, 0, 0, 1 ]])

# # TODO: should we use math.sin and math.cos instead of numpy versions?

# def _Rx(angle):
#     """Creates a rotation matrix about the x-axis"""
#     return _Rxsc(np.sin(angle), np.cos(angle))

# def _Rysc(s, c):
#     """Create a rotation matrix about the y-axis

#     This version takes a sin and cos value as arguments and populates
#     the appropriate parts of a 4x4 matrix.
#     """
#     return np.asarray([
#         [   c, 0.0,   s, 0.0 ],
#         [ 0.0, 1.0, 0.0, 0.0 ],
#         [  -s, 0.0,   c, 0.0 ],
#         [ 0.0, 0.0, 0.0, 1.0 ]])

# def _Ry(angle):
#     """Creates a rotation matrix about the y-axis"""
#     return _Rysc(np.sin(angle), np.cos(angle))

# def _Rzsc(s, c):
#     """Create a rotation matrix about the z-axis

#     This version takes a sin and cos value as arguments and populates
#     the appropriate parts of a 4x4 matrix.
#     """
#     return np.asarray([
#         [   c,  -s, 0.0, 0.0 ],
#         [   s,   c, 0.0, 0.0 ],
#         [ 0.0, 0.0, 1.0, 0.0 ],
#         [ 0.0, 0.0, 0.0, 1.0 ]])

# def _Rz(angle):
#     """Creates a rotation matrix about the z-axis"""
#     return _Rzsc(np.sin(angle), np.cos(angle))

# def _T(x, y, z):
#     """Creates a translation matrix"""
#     return np.asarray([
#         [ 1.0, 0.0, 0.0,   x ],
#         [ 0.0, 1.0, 0.0,   y ],
#         [ 0.0, 0.0, 1.0,   z ],
#         [ 0.0, 0.0, 0.0, 1.0 ]])

_Ra1 = rotationXsc(1, 0)
_Ra5 = rotationXsc(-1, 0)

# DH parameters (Raw)
_d1 =  0.089159
_a2 = -0.42500
_a3 = -0.39225
_d4 =  0.10915
_d5 =  0.09465
_d6 =  0.08230

# This is the center of the gripper.  Set to 0 to match UR5's tool point.
# _ee_offset = 0.129459

# UR5 tool point
# _ee_offset = 0

# tip of gripper
# _ee_offset = 0.18

# Transforms used in the forward kinmatics
_Td1 = translation(  0, 0, _d1)
_Ta2 = translation(_a2, 0, 0)
_Ta3 = translation(_a3, 0, 0)
_Td4 = translation(  0, 0, _d4)
_Td5 = translation(  0, 0, _d5)
_Td6 = translation(  0, 0, _d6)
_RT3 = _Ta3 @ _Td4
_RT4 = _Ra1 @ _Td5
_RT5 = _Ra5 @ _Td6

calls = 0


def parse_ee_offset(ee_offset):
    """Different offset locations for the EE"""
    if isinstance(ee_offset, str):
        if ee_offset == 'gripper_center':
            ee_offset = 0.16
        elif ee_offset == 'gripper_tip':
            ee_offset = 0.18
        elif ee_offset == 'tool_point':
            ee_offset = 0.0
        else:
            raise ValueError('Invalid string')

    assert isinstance(ee_offset, float)

    return ee_offset

class UR5RobotKinematics:
    """UR5 Robot kinematic calculations
    
    This class represents the kinematic state of a UR5 robot.  It has a
    6 element configuration that stores the joint angles (in radians),
    the coordinate frames of the joints (as 4x4 matrices).
    """

    # AKA degrees of freedom
    DIMENSIONS = 6

    MIN_CONFIG = np.array([ -2*pi, -pi, -2*pi, -2*pi, -2*pi, -2*pi ])
    MAX_CONFIG = np.array([  2*pi,   0,  2*pi,  2*pi,  2*pi,  2*pi ])

    MAX_VELOCITY = np.array([3.]*6)
    MIN_VELOCITY = -MAX_VELOCITY

    MAX_ARM_EFFORT = 150.
    MAX_WRIST_EFFORT = 28.

    # UR5 has a payload limit of 5 kg
    MASS_LIMIT = 5.

    # Compute an approximation of the maximum acceleration based on
    # holding the maximum payload at the maximum extension
    MAX_ACCELERATION = np.array([
        MAX_ARM_EFFORT / (MASS_LIMIT * 1.25), # shoulder pan
        MAX_ARM_EFFORT / (MASS_LIMIT * 1.25), # shoulder lift
        MAX_ARM_EFFORT / (MASS_LIMIT * 0.75), # elbow
        MAX_WRIST_EFFORT / (MASS_LIMIT * 0.4), # wrist 1
        MAX_WRIST_EFFORT / (MASS_LIMIT * 0.3), # wrist 2
        MAX_WRIST_EFFORT / (MASS_LIMIT * 0.2) # wrist 3
        ])

    MAX_JERK = MAX_ACCELERATION * 10.
        
    def __init__(self, config=np.zeros(6), base=np.eye(4), ee_offset='gripper_center'):
        """Initializes the robot with an optional in initial configuration and base transform"""
        self.config = config
        self.base = base
        self.joint_origins = [np.eye(4)] * 6

        self.ee_offset = parse_ee_offset(ee_offset)
        self._Tee = translation(0, 0, self.ee_offset)

        self.set_config(self.config)

    def set_config(self, q):
        """Sets the configuration of the robot

        This updates the internal coordinate frames and must be called"""
        self.config = q
        # compute the forward kinematics
        self.joint_origins[0] = self.base                         @ _Td1
        self.joint_origins[1] = self.joint_origins[0] @ rotationZ(q[0]) @ _Ra1
        self.joint_origins[2] = self.joint_origins[1] @ rotationZ(q[1]) @ _Ta2
        self.joint_origins[3] = self.joint_origins[2] @ rotationZ(q[2]) @ _RT3 # @ _Ta3        @ _Td4
        self.joint_origins[4] = self.joint_origins[3] @ rotationZ(q[3]) @ _RT4 # @ _Ra1 @ _Td5
        self.joint_origins[5] = self.joint_origins[4] @ rotationZ(q[4]) @ _RT5 # @ _Ra5 @ _Td6
        self.ee_frame         = self.joint_origins[5] @ rotationZ(q[5]) @ self._Tee

    @staticmethod
    def random_config(rng):
        return rng.uniform(UR5Robot.MIN_CONFIG, UR5Robot.MAX_CONFIG)
        # return np.array([rng.uniform(UR5Robot.MIN_CONFIG[i], UR5Robot.MAX_CONFIG[i])
        #                      for i in range(UR5Robot.DIMENSIONS)])
        
    def compute_jacobian(self):
        """Computes the Jacobian for the end-effector frame

        In the Jacobian, each column corresponds to a joint in the same
        order as the configuration vector, and each row corresponds to
        the change in tx, ty, tz, rx, ry, rz.  Thus:

        dtx/dq0 dtx/dq1 ... dtx/dq5
        dty/dq0 dty/dq1 ... dty/dq5
          ...     ...         ...
        drz/dq0 drz/dq1 ... drz/dq5
        """
        cols = []
        et = self.ee_frame[0:3,3] # translation of end effector
        for i in range(6):
            jz = self.joint_origins[i][0:3,2] # linear z column = (R * axis) = R * [0 0 1]^T
            jt = self.joint_origins[i][0:3,3] # translation at frame
            cols.append(np.concatenate([np.cross(jt - et, jz), jz]))
        return np.column_stack(cols)

    @staticmethod
    def ika8(target, ee_offset='gripper_center'):
        global calls
        calls += 1
        """Computes an analytic inverse kinematic.

        Every target frame has 8 possible configurations.  This method
        will return all of them."""

        # algorithm adapted from
        # https://github.com/mc-capolei/python-Universal-robot-kinematics/blob/master/universal_robot_kinematics.py
        
        d6ee = _d6 + parse_ee_offset(ee_offset)

        th = np.zeros([6, 8])

        p05 = target[0:3,2] * -d6ee + target[0:3,3] 

        psi = math.atan2(p05[1], p05[0])
        phi = math.acos(_d4 / math.sqrt(np.sum(p05[0:2]**2)))

        th[0,0:4] = pi/2 + psi + phi
        th[0,4:8] = pi/2 + psi - phi

        i = 0
        while i<8:
            # does it matter if rotationXsc is (0, -1) or (-1, 0)
            t10 = rotationXsc(-1, 0) @ rotationZ(-th[0,i]) @ translation(0, 0, -_d1) #PLAY WITH THIS
            # t10 = rotationXsc(-1, 0) @ rotationZ(-th[0,i]) @ translation(0, 0, -_d1) #ORIGINAL
            t16 = t10 @ target
            a = math.acos((t16[2,3] - _d4) / d6ee)
            th[4, i  :i+2] =  a
            th[4, i+2:i+4] = -a

            at61 = math.atan2(-t16[2,1], t16[2,0])
            th[5, i  :i+2] = at61
            th[5, i+2:i+4] = (at61 + pi) if at61 < 0 else (at61 - pi) #PLAY WITH THIS
            # th[5, i+2:i+4] = (at61 + pi) if at61 < 0 else (at61 - pi) #ORIGINAL
            
            j=i+4
            while i<j:
                t14 = t16 @ rotationZ(-th[5,i]) @ translation(0,0,-d6ee) @ rotationXsc(1, 0) @ rotationZ(-th[4,i]) @ translation(0,0,-_d5) # extra rotationXsc?
                p13 = t14[0:2,3] - t14[0:2,1] * _d4
                p13norm = np.sum(p13**2) #is this the actual squared norm
                aarg = (p13norm - _a2*_a2 - _a3*_a3) / (2 * _a2 * _a3)
                if abs(aarg) <= 1:
                    t3 = math.acos((p13norm - _a2*_a2 - _a3*_a3) / (2 * _a2 * _a3))
                else:
                    t3 = math.nan
                
                th[2, i  ] = t3
                th[2, i+1] = -t3
                
                at13 = math.atan2(p13[1], -p13[0])
                asin3 = math.asin(_a3 * math.sin(th[2,i])/math.sqrt(p13norm))
                th[1,i  ] =  asin3 - at13
                th[1,i+1] = -asin3 - at13
                k = i+2
                while i<k:
                    t34 = translation(-_a3,0,0) @ rotationZ(-th[2,i]) @ translation(-_a2, 0, 0) @ rotationZ(-th[1,i]) @ t14
                    th[3,i] = math.atan2(t34[1,0], t34[0,0])
                    i += 1
        return th

    def ik(self, target, ee_offset='gripper_center'):
        """Computes the inverse kinematic to the target frame.

        Of the possible IK frames, the closest to the robot's current
        configuration is selected.

        Returns True on success, otherwise False if there are no IK
        solutions (e.g., the frame is out of range)
        """

        allik = self.ika8(target,ee_offset)
#         print(allik)
        b = math.inf
        j = -1
        for i in range(8):
            # computes the SO(2) distance between each joint
#             import pdb;pdb.set_trace()
            s = np.abs(self.config - allik[:,i])
            d = np.sum(s)
            # s = np.fmod(s, pi*2)
            # d = np.sum(np.min(pi*2 - s))
            if d < b and not math.isnan(d):
                b = d
                j = i

        # this can happen if all values are nan
        if j == -1:
            return False

        self.set_config(allik[:,j])
        return True
    
    def ikmod(self, target, ee_offset='gripper_center'):
        """Computes the inverse kinematic to the target frame.

        Of the possible IK frames, the closest to the robot's current
        configuration is selected.

        Returns True on success, otherwise False if there are no IK
        solutions (e.g., the frame is out of range)
        """

        allik = self.ika8(target,ee_offset)
#         print(allik)
        b = math.inf
        j = -1
        for i in range(8):
            # computes the SO(2) distance between each joint
#             import pdb;pdb.set_trace()
            s = np.abs(self.config - mod_range(allik[:,i]))
            d = np.sum(s)
#             s = np.fmod(s, pi*2)
#             d = np.sum(np.min(pi*2 - s))
            if d < b and not math.isnan(d):
                b = d
                j = i

        # this can happen if all values are nan
        if j == -1:
            return False

        self.set_config(mod_range(allik[:,j]))
        return True
        
