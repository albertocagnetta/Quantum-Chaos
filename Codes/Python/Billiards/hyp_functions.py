import numpy as np
from numpy import sign, dot, multiply, cos, sin, real, imag, matmul, subtract, linspace, float64
# from matplotlib.animation import FuncAnimation

from mpmath import kleinj
import math
from math import sqrt, acos, tan, atan
from manim import *
import string
import numpy.linalg as nl
from scipy import interpolate
from scipy.interpolate import interp1d


# CONSTANTS
INF = float('inf')
EPS = 1e-12
EPSX = 1e-12


def norm(v):
    return sqrt(np.dot(v, v))


def angle(v): #gives the angle of a vector
    if v[0] == 0:
        if v[1] > 0:
            theta=PI/2
        else:
            theta=-PI/2
    elif v[1] == 0:
        if v[0] > 0:
            theta=0
        else:
            theta=PI
    elif v[0] > 0:
        theta = atan(v[1]/v[0])
    else:
        theta = atan(v[1]/v[0])-PI*np.sign(v[1]/v[0])
    return theta



def compl_to_2D(z):
    return np.array([np.real(z), np.imag(z)])


def D2_to_compl(p):
    return p[0] + p[1] * 1j


def from_C_to_sphere(z, r):
    return np.multiply(r / (np.real(z) ** 2 + np.imag(z) ** 2 + r ** 2),
                       [2 * r * np.real(z), 2 * r * np.imag(z), np.real(z) ** 2 + np.imag(z) ** 2 - r ** 2])


def hyp_action(M, z):
    return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])


def fixpt(M):  # Computing the fixed point of a 2X2 matrix in the upper half plane
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    if c == 0:
        return INF
    else:
        return (-(d - a) + np.sign(c) * 1J * sqrt(-(d - a) ** 2 - 4 * b * c)) / (2 * c)


def geod_H(z, w):  # z,w are complx numbers
    if z == INF:
        return [np.real(w), INF]
    elif w == INF:
        return [np.real(z), INF]
    else:
        z1 = compl_to_2D(z)
        w1 = compl_to_2D(w)
        if z1[0] == w1[0]:
            return [z1[0], INF]
        else:
            center = (w1[1] ** 2 - z1[1] ** 2 + w1[0] ** 2 - z1[0] ** 2) / (2 * (w1[0] - z1[0]))
            return [center, norm(z1 - np.array([center, 0]))]  # the second argument is the radius

def geod_H_angles(z, w):  # z,w are complx numbers
    [c, r]=geod_H(z, w)
    if r == INF:
        return [np.imag(z), np.imag(w)]
    else:
        z1 = compl_to_2D(z)
        w1 = compl_to_2D(w)
        theta1 = angle(z1-np.array([c, 0]))
        theta2 = angle(w1-np.array([c, 0]))
        return [theta1, theta2]  # the second argument is the radius



def matrix_direction(p, v):
    # given a point and a direction, gives the matrix of the correspondent geodesic
    # print("punto input:",p)
    theta = 1 / 2 * (PI / 2 - angle(v))
    # print("punti p0 e p1:",p[0],p[1])
    A = np.array([[math.sqrt(p[1]), (p[0]) / math.sqrt(p[1])], [0, 1 / math.sqrt(p[1])]])
    K = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(A, K)


def geod_from_matrix(M):
    # gives the geometric elements of the geodesic from the matrix: the starting point, end point, center of circle and radius
    if M[1, 0] == 0:
        return [M[0, 1] / M[1, 1], INF, INF, INF]
    elif M[1, 1] == 0:
        return [INF, M[0, 0] / M[1, 0], INF, INF]
    else:  # warning: the first column gives the end point, the second gives the starting one (ad-bc=1>0)
        return [
            M[0, 1] / M[1, 1], M[0, 0] / M[1, 0], (M[0, 0] / M[1, 0] + M[0, 1] / M[1, 1]) / 2,
            np.abs((M[0, 1] / M[1, 1] - M[0, 0] / M[1, 0]) / 2)
        ]


def geod_from_point(p, v):
    # gives the geometric elements of the geodesic from the matrix: the starting point, end point, center of circle and radius
    if v[0] == 0:
        return [p[0],INF]
    else:
        c = p[0] + p[1] * v[1] / v[0]
        r = norm(np.subtract(p, [c, 0]))
    return [c, r]




def inters_geodesic_and_circled_wall(p, v, C, R):  # C is the center of the circled wall
    # print("dati geod and circle:", a,b,c,r)
    if v[0] == 0:
        if np.abs(C - p[0]) >= R:
            return [[INF, INF], [INF, INF]]
        elif v[1] > 0 and norm(p) < R:
            x_int = p[0]
            y_int = math.sqrt(R ** 2 - (x_int - C) ** 2)
            return [[x_int, y_int], v]
        elif v[1] < 0 and norm(p) > R:
            x_int = p[0]
            y_int = math.sqrt(R ** 2 - (x_int - C) ** 2)
            return [[x_int, y_int], v]
        else:
            return [[INF, INF], [INF, INF]]
    else:
        # center of circle of geodesic
        c = p[0] + p[1] * v[1] / v[0]
        r = norm(np.subtract(p, [c, 0]))
        if np.abs(C - c) > R + r:
            return [[INF, INF], [INF, INF]]
        elif C - R < c - r and c + r < C + R:
            return [[INF, INF], [INF, INF]]
        elif c - r < C - R and C + R < c + r:
            return [[INF, INF], [INF, INF]]
        else:
            x_int = (r ** 2 - R ** 2 + C ** 2 - c ** 2) / (2 * (C - c))
            y_int = math.sqrt(r ** 2 - (x_int - c) ** 2)
            v = np.subtract([x_int, y_int], [c, 0])
            v = np.multiply(1 / r, v)
            v = np.dot([[0, 1], [-1, 0]], v)
            return [[x_int, y_int], v]


def inters_geodesic_and_line_wall(p, v, X):  # C is the center of the circled wall
    # print("dati geod and wall:", a, b, c, r)
    # if p[0] == X:
    if v[0] == 0:
        return [[INF, INF], [INF, INF]]
    else:
        c = p[0] + p[1] * v[1] / v[0]
        r = norm(np.subtract(p, [c, 0]))
        if np.abs(c - X) > r:
            return [[INF, INF], [INF, INF]]
        else:
            x_int = X
            y_int = math.sqrt(r ** 2 - (X - c) ** 2)
            v = np.subtract([x_int, y_int], [c, 0])
            v = np.multiply(1 / r, v)
            v = np.dot([[0, 1], [-1, 0]], v)
            return [[x_int, y_int], v]




def hyp_action_vel(p, v, M):  # returns the hyp action on the velocity vector
    theta = angle(v)
    tau = np.cos(theta) + np.sin(theta) * 1j
    new_tau = tau / (M[0, 1] * (p[0] + p[1] * 1j) + M[1, 1]) ** 2
    return compl_to_2D(new_tau)


def starting_from_M(M):
    [a, b, c, r] = geod_from_matrix(M)
    p = [0, math.sqrt(r ** 2 - c ** 2)]
    v = np.subtract(p, [c, 0])
    v = np.multiply(1 / norm(v), v)
    v = np.dot([[0, 1], [-1, 0]], v)
    return [p, v]


def points_on_geodesic(p1, p2, c):
    theta1 = angle(np.subtract(p1, c))
    theta2 = angle(np.subtract(p2, c))
    r = norm(np.subtract(p1, c))
    return [[r * np.cos(a) + c[0], r * np.sin(a) + c[1]] for a in linspace(theta1, theta2, 20)]


