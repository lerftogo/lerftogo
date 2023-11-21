import numpy as np
from numpy import pi
import math

# class Transform:
#     def __init__(self, m = np.eye(4)):
#         self.matrix = m

#     def __matmul__(self, other):
#         return Transform(self.matrix @ other.matrix)

#     def linear(self):
#         return self.matrix[0:3, 0:3]

#     def translation(self):
#         return self.matrix[0:3, 3]



def rotationXsc(s, c):
    """Create a rotation matrix about the x-axis

    This version takes a sin and cos value as arguments and populates
    the appropriate parts of a 4x4 matrix.
    """
    return np.asarray([
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0,   c,  -s, 0.0 ],
        [ 0.0,   s,   c, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ]])

def rotationX(angle):
    """Creates a rotation matrix about the x-axis"""
    return rotateXsc(math.sin(angle), math.cos(angle))

def rotationYsc(s, c):
    """Create a rotation matrix about the y-axis

    This version takes a sin and cos value as arguments and populates
    the appropriate parts of a 4x4 matrix.
    """
    return np.asarray([
        [   c, 0.0,   s, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [  -s, 0.0,   c, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ]])

def rotationY(angle):
    """Creates a rotation matrix about the y-axis"""
    return rotationYsc(math.sin(angle), math.cos(angle))

def rotationZsc(s, c):
    """Create a rotation matrix about the z-axis

    This version takes a sin and cos value as arguments and populates
    the appropriate parts of a 4x4 matrix.
    """
    return np.asarray([
        [   c,  -s, 0.0, 0.0 ],
        [   s,   c, 0.0, 0.0 ],
        [ 0.0, 0.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ]])

def rotationZ(angle):
    """Creates a rotation matrix about the z-axis"""
    return rotationZsc(math.sin(angle), math.cos(angle))

def translation(x, y, z):
    """Creates a translation matrix"""
    return np.asarray([
        [ 1.0, 0.0, 0.0,   x ],
        [ 0.0, 1.0, 0.0,   y ],
        [ 0.0, 0.0, 1.0,   z ],
        [ 0.0, 0.0, 0.0, 1.0 ]])


def get_translation(m):
    return m[0:3, 3]

def get_linear(m):
    return m[0:3, 0:3]
