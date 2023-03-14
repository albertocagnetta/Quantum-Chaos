import numpy as np
import sympy as sp
from sympy import cos, sin, Symbol, diff, div, Abs, sqrt
from math import sqrt, acos, tan, atan, pi
from scipy import sparse
from scipy.sparse import linalg as sla
from numpy import sin, cos, pi, linspace, sign, dot, multiply, add, roots
import numpy.linalg as nl


PI = pi

INF = 1000
EPS = 1e-12
t = Symbol('t') #setting the variable
x = Symbol('x')
y = Symbol('y')

def norm(v):
    "norm of a vector v"
    return sqrt(np.dot(v, v))

def angle(v): #
    "gives the angle of a 2D-vector"
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

def time(p1, p2, velocity):
    " computes the distance done and the time of the path between two points "
    return norm(np.subtract(p1, p2)) / velocity

############################################################
"Bouncing functions"

def bounce_circle(direction, b_point, center):
    """
    Calculate the new direction after a bouncing with a circle wall
    b_point: 2D vector of coordinates of the impact point between the particle and the circle wall
    direction: 2D vector which is the velocity of the particle
    center: center of the circle
    """
    F1 = [[b_point[0]-center[0], b_point[1]-center[1]], [b_point[1]-center[1], center[0]-b_point[0]]]
    F2 = [[-1, 0], [0, 1]]
    F3 = [[center[0]-b_point[0],center[1]-b_point[1]],[-(b_point[1]-center[1]),b_point[0]-center[0]]]
    #F3 = [[b_point[0]-center[0],b_point[1]-center[1]],[-b_point[1]+center[1],-b_point[0]+center[0]]]
    F3 = -np.dot(1/((b_point[0]-center[0])**2+(b_point[1]-center[1])**2), F3)
    return np.dot(np.matmul(np.matmul(F1, F2), F3), direction)


def bounce_line(direction, perp):
    """
    Calculate the new direction after a bouncing with a line wall
    b_point: 2D vector of coordinates of the impact point between the particle and the circle wall
    direction: 2D vector which is the velocity of the particle
    perp: 2D vector perpendicular to the line (e.g. if the line has equation ax+by=c, then perp=(a,b))
    """
    return direction - 2 * np.dot(direction, perp)/np.dot(perp,perp)*perp


def bounce_polar(direction, b_point, polar):
    """
        Calculate the new direction after a bouncing with a wall described by a polar equation
        b_point: 2D vector of coordinates of the impact point between the particle and the circled wall
        direction: 2D vector which is the velocity of the particle
        polar: polar equation of the boundary, in variable 't'
    """
    vx_coord = diff(polar * sp.cos(t), t)
    vy_coord = diff(polar * sp.sin(t), t)

    w = [vx_coord.subs(t, angle(b_point)), vy_coord.subs(t,angle(b_point))]
    M1 = [[w[0], w[1]], [w[1], -w[0]]]
    M2 = [[1, 0], [0, -1]]
    M3 = [[-w[0], -w[1]], [-w[1], w[0]]]
    M3 = -np.dot(1 / (w[0] ** 2 + w[1] ** 2), M3)
    return np.dot(np.matmul(np.matmul(M1, M2), M3), direction)


#####################################################################################
"Intersection functions"


def circle_intersection(p, v, center, R, angle1, range, region):
    """
    Intersection between our point and a circle wall (considering only the arc between angles angle1 and angle1+range)
    :param p: point coordinates
    :param v: velocity
    :param center: center of the circle
    :param R: radius of the circle
    :param angle1: first angle of the circle (from -PI to PI)
    :param range: range of the circle arc (from 0 to 2*PI) (counterclockwise)
    :param region: 1 if the point bounce on the inside of the circle, -1 otherwise
    :return: the time of impact of the point with the circle wall
    """
    test_p = p + np.multiply(np.dot(np.subtract(center, p), v), v)
    if norm(test_p-center) > R:
        #print("test 1")
        return INF
    #elif np.dot(np.subtract(cen, center), v) < 0:
    #    #print("test 2")
    #    return INF
    else:
        t = np.dot(np.subtract(center, p), v) + region * np.sin(acos(norm(test_p-center)/R))*R
        if t < 0 :
            return INF
        else:
            new_p = p + np.multiply(t, v)
            if (angle(np.subtract(new_p,center)) >= angle1 and angle(np.subtract(new_p,center)) <= angle1+range) or (angle(np.subtract(new_p,center))+2*PI >= angle1 and angle(np.subtract(new_p,center))+2*PI <= angle1+range):
                return t
            else:
                #print("test 3")
                return INF


def line_intersection(p,v,abc,x1,x2):
    """
    :param p: point coordinates
    :param v: velocity vector
    :param abc: vector (a,b,c) of equation ax+by=c (the semiplane of the billiard is ax+by<c)
    :param x1: first vertex of segment wall
    :param x2: second vertex of segment wall
    :return: time of intersection
    ! ONLY x1 CAN BE A NULL VECTOR (0,0)!
    """
    # bisection
    perp = np.array([abc[0], abc[1]])
    if nl.det(np.array([[v[0],-abc[1]],[v[1],abc[0]]])) == 0: # line and velocity parallel
        print("test 1")
        return INF
    else:
        # l, r = 0, 1000
        # while r-l > EPS:
        #     m = (l+r)/2
        #     if np.dot(perp, (p+np.multiply(m, v))) > abc[2]:
        #         r = m
        #     else:
        #         l = m
        l = (abc[2]-np.dot(p,perp))/np.dot(v,perp)
        new_p = p + np.multiply(l, v)
        #print(new_p)
        if norm(x1) == 0:
            check = np.array([new_p[0]*x2[0],new_p[1]*x2[1]])
        else:
            check = nl.inv(np.array([[x1[0],x2[0]],
                                      [x1[1],x2[1]]
                                      ])).dot(new_p)
        #print(check)
        if check[0]>=0 and check[1]>=0 and l > EPS:
            print("test 2")
            return l
        else:
            return INF

def polar_wall_intersection(p, v, boundary):
    w = np.multiply(1/norm(v), v)
    new_p = p
    EPSt = 1e-12
    #computing the coefficient from the algebraic equation of the boundary
    # x = p[0]+v[0]*t
    # y = p[1]+v[1]*t
    new_bound = boundary.subs(x, new_p[0] + w[0] * t)
    new_bound = new_bound.subs(y, new_p[1] + w[1] * t)
    # now let's get the coefficients
    coeff = []
    for i in range(5):
        coeff.insert(0, new_bound.subs(t, 0))
        new_bound = div(new_bound - new_bound.subs(t, 0), t, domain='RR')[0]  #we want only the quotient, not the reminder
    # now the list has the coefficient of the polynomial
    #times = np.roots(list)
    #print("%%%%%%%%%%%%%%%  BOING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #print(t_val)
    if len(list(filter(lambda a: np.imag(a) == 0 and np.real(a) > EPSt, np.roots(coeff)))) == 0:
        print("tempi problematici:", np.roots(coeff))
        return np.real(max(filter(lambda a: np.imag(a) == 0 and np.real(a) < -EPSt, np.roots(coeff))))
    else:
        return np.real(min(filter(lambda a: np.imag(a) == 0 and np.real(a) > EPSt, np.roots(coeff))))
