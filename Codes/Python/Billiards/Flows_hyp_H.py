# from sage.all import *
import numpy as np
from manim import *
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
import math
from math import sqrt, acos, tan
from mpmath import kleinj
from numpy import sin, cos, pi, linspace, abs, sign, subtract, random
import numpy.linalg as nl

INF = float('inf')
SCALE = 1


# Sc = np.matrix([[np.sqrt(SCALE), 0], [0, np.sqrt(SCALE)**(-1)]])


def compl_to_2D(z):
    return np.array([np.real(z), np.imag(z)])


def D2_to_compl(p):
    return p[0] + p[1] * 1j


def norm(v):
    return math.sqrt(np.dot(v, v))


def angle(v):  # gives the angle of a vector
    if v[0] == 0:
        if v[1] > 0:
            theta = PI / 2
        else:
            theta = -PI / 2
    elif v[1] == 0:
        if v[0] > 0:
            theta = 0
        else:
            theta = PI
    elif v[0] > 0:
        theta = math.atan(v[1] / v[0])
    else:
        theta = math.atan(v[1] / v[0]) - PI * np.sign(v[1] / v[0])
    return theta


def fixpt(M):  # Computing the fixed point of a 2X2 matrix in the upper half plane
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    if c == 0:
        return INF
    else:
        return (-(d - a) + np.sign(c) * 1J * sqrt(np.abs(-(d - a) ** 2 - 4 * b * c))) / (2 * c)


def hyp_action(M, z):
    if z == INF:
        if M[0, 1] == 0:
            return INF
        else:
            return M[0, 0] / M[0, 1]
    else:
        return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])

def matrix_direction(p, v):
    #given a point and a direction, gives the matrix of the correspondent geodesic
    #print("punto input:",p)
    theta = 1/2*(PI/2-angle(v))
    #print("punti p0 e p1:",p[0],p[1])
    A = np.array([[math.sqrt(p[1]), (p[0])/math.sqrt(p[1])], [0, 1/math.sqrt(p[1])]])
    K = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(A, K)

def geod_from_matrix(M):
    # gives the geometric elements of the geodesic from the matrix: the starting point, end point, center of circle and radius
    if M[1][0] == 0:
        return [M[0][1]/M[1][1], INF, INF, INF]
    elif M[1][1] == 0:
        return [INF, M[0][0]/M[1][0], INF,INF]
    else: # warning: the first column gives the end point, the second gives the starting one (ad-bc=1>0)
        return [
            M[0][1]/M[1][1], M[0][0]/M[1][0], (M[0][0]/M[1][0]+M[0][1]/M[1][1])/2, np.abs((M[0][1]/M[1][1]-M[0][0]/M[1][0])/2)
        ]


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
            c = (w1[1] ** 2 - z1[1] ** 2 + w1[0] ** 2 - z1[0] ** 2) / (2 * (w1[0] - z1[0]))
            return [c, norm(z1 - np.array([c, 0]))]


def draw_geod(z, w, color, plane):
    # z = hyp_action(Sc, z)
    # w = hyp_action(Sc, w)
    c, r = geod_H(z, w)
    if r == INF:
        if z == INF:
            # return Line([c, np.imag(w), 0], [c, 100, 0], stroke_width=2, color=color)
            return Line(plane.n2p(c + 1j * np.imag(w)), plane.n2p(c + 1j * 100), stroke_width=2, color=color)
        elif w == INF:
            # return Line([c, np.imag(z), 0], [c, 100, 0], stroke_width=2, color=color)
            return Line(plane.n2p(c + 1j * np.imag(z)), plane.n2p(c + 1j * 100), stroke_width=2, color=color)
        else:
            # return Line([c, np.imag(z), 0], [c, np.imag(w), 0], stroke_width=2, color=color)
            return Line(plane.n2p(c + 1j * np.imag(z)), plane.n2p(c + 1j * np.imag(w)), stroke_width=2, color=color)
    else:
        theta1 = angle(np.subtract(compl_to_2D(z), [c, 0]))
        theta2 = angle(np.subtract(compl_to_2D(w), [c, 0]))
        return Arc(r * SCALE, theta1, theta2 - theta1, arc_center=plane.n2p(c), stroke_width=2, color=color)


def draw_TriFund_star(A, B,
                      plane):  # Draw the fundamental domain of a triangle group with generators A,B of finite orders. No middle line, fixed points of A, B are marked by o and * respectively
    C = B @ A
    # L=[A,B,C]
    # P=[fixpt(A),fixpt(B),fixpt(C),fixpt(A*C*A**(-1))]
    vert = [fixpt(A), fixpt(B), fixpt(C), fixpt(A @ C @ nl.inv(A))]
    print(vert)
    # vert = [hyp_action(Sc,a) for a in vert]
    # print(vert)
    # vert = [compl_to_2D(a) for a in vert]
    list = []
    list.append(draw_geod(vert[0], vert[1], RED, plane))
    list.append(draw_geod(vert[1], vert[3], GREEN, plane))
    list.append(draw_geod(vert[0], vert[3], YELLOW, plane))
    list.append(draw_geod(vert[1], vert[2], ORANGE, plane))
    list.append(draw_geod(vert[0], vert[2], PINK, plane))
    list.append(draw_geod(vert[2], vert[3], TEAL_B, plane))
    # d1 = Dot([compl_to_2D(vert[0])[0], compl_to_2D(vert[0])[1], 0], color=YELLOW)
    # d2 = Dot([compl_to_2D(vert[1])[0], compl_to_2D(vert[1])[1], 0], color=RED)
    # d3 = Dot([compl_to_2D(vert[2])[0], compl_to_2D(vert[2])[1], 0], color=GREEN)
    # d4 = Dot([compl_to_2D(vert[3])[0], compl_to_2D(vert[3])[1], 0], color=ORANGE)
    d1 = Dot(plane.n2p(vert[0]), color=YELLOW)
    d2 = Dot(plane.n2p(vert[1]), color=RED)
    d3 = Dot(plane.n2p(vert[2]), color=GREEN)
    d4 = Dot(plane.n2p(vert[3]), color=ORANGE)
    list.append(d1)
    list.append(d2)
    list.append(d3)
    list.append(d4)
    # list.append(MathTex("fix A").next_to(d1, UR, 0.1))
    # list.append(MathTex("fix B").next_to(d2, UR, 0.1))
    # list.append(MathTex("fix C").next_to(d3, UR, 0.1))
    return list


def draw_TriFund2_star(A, B,
                       plane):  # Draw the fundamental domain of a triangle group with generators A,B of finite orders. No middle line, fixed points of A, B are marked by o and * respectively
    C = B @ A
    # L=[A,B,C]
    # P=[fixpt(A),fixpt(B),fixpt(C),fixpt(A*C*A**(-1))]
    vert = [fixpt(A), fixpt(B), fixpt(C), fixpt(nl.inv(C) @ B @ C)]
    print(vert)
    list = []
    list.append(draw_geod(vert[0], vert[1], RED, plane))
    list.append(draw_geod(vert[1], vert[3], GREEN, plane))
    list.append(draw_geod(vert[0], vert[3], YELLOW, plane))
    list.append(draw_geod(vert[1], vert[2], ORANGE, plane))
    list.append(draw_geod(vert[0], vert[2], PINK, plane))
    list.append(draw_geod(vert[2], vert[3], TEAL_B, plane))
    d1 = Dot(plane.n2p(vert[0]), color=YELLOW)
    d2 = Dot(plane.n2p(vert[1]), color=RED)
    d3 = Dot(plane.n2p(vert[2]), color=GREEN)
    d4 = Dot(plane.n2p(vert[3]), color=ORANGE)
    list.append(d1)
    list.append(d2)
    list.append(d3)
    list.append(d4)
    return list


def val(s):
    return [2 * np.cos(2 * PI / s), 2 * np.sin(2 * PI / s)]


def gen_trian_group(a, b, c):  # gives three matrix generator for triangle group with angle a,b,c with finite orders
    if a == INF:
        lama, mua = 2, 0
    else:
        lama, mua = val(2 * a)
    if b == INF:
        lamb, mub = 2, 0
    else:
        lamb, mub = val(2 * b)
    if c == INF:
        lamc, muc = 2, 0
    else:
        lamc, muc = val(2 * c)
    A = np.matrix([[lama / 2, mua / 2], [-mua / 2, lama / 2]])
    x = (lama * lamb + 2 * lamc) / (mua * mub)
    t = x + np.sqrt(x ** 2 - 1)
    B = np.matrix([[lamb / 2, t * mub / 2], [-mub / 2 / t, lamb / 2]])
    M = np.matrix([[1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)]])
    A = nl.inv(M) @ A @ M
    B = nl.inv(M) @ B @ M
    return [A, B, B @ A]


def orbita(F, x, N):  # questo comando orbita è modificato e fornisce l'orbita dell'azione di una matrice su un complesso
    X = [np.array([x, 1])]  # primo punto dell'orbita, creiamo una lista
    for n in range(N - 1):
        y = np.dot(F, X[-1])
        X.append(y / y[1])
    return X


################## FARE UNA FUNZIONE UNICA, CHE FA COSE DIVERSE IN BASE ALLA TRACCIA

def hyp_action_H(p,M):  #c_fixed and r_fixed are the euclidian center and ray of the fixed semicircle
    c_fixed = -(M[1,1]-M[0,0])/(2*M[1,0])
    r_fixed = np.sqrt((M[1,1]+M[0,0])**2-4)/(2*np.abs(M[1,0]))
    a = c_fixed - r_fixed
    b = c_fixed + r_fixed
    z = D2_to_compl(p)
    c = ((b-z)*np.abs(a)**2+(z-a)*np.abs(b)**2+(a-b)*np.abs(z)**2)/( (b-z)*np.conj(a)+(z-a)*np.conj(b)+(a-b)*np.conj(z) )
    c = compl_to_2D(c)
    r = norm(np.subtract(p,c))
    return [c,r,c_fixed,r_fixed]


#def parab_action_H(p,M):


def ellip_action_H(p,M): #returns the euclidian circle and ray of the circle orbit of p under M (elliptic)
    q = fixpt(M)
    q = compl_to_2D(q)
    r_hyp = np.arccosh(
        1+( (p[0]-q[0])**2+(p[1]-q[1])**2 )/(2*p[1]*q[1])
    )
    a = np.array([q[0],q[1]*np.exp(-r_hyp)])
    b = np.array([q[0],q[1]*np.exp(r_hyp)])
    c = np.multiply(0.5,a+b)
    r = norm(c-a)
    return [c,r]


def parab_action_H(p,M):
    q = fixpt(M) #q is one real point
    q = np.real(q)
    q = compl_to_2D(q)
    t = (q[0] - p[0])/2/(p[1]-q[1])
    c = np.multiply(0.5, q+p)+np.multiply(t,np.dot(np.array([[0,1],[-1,0]]),(p-q)))
    r = c[1]
    print("centro e raggio e punto fisso:",c,norm(p-c),q)
    return [c, r, q]




def uns_horo_flow(p,v):
    A = matrix_direction(p,v)
    [a,b,c,r] = geod_from_matrix(A)
    theta =angle(np.subtract(p,[c,0]))
    alpha = (PI-theta)/2
    return [a,r*np.tan(alpha)]

def stb_horo_flow(p,v):
    A = matrix_direction(p,v)
    [a,b,c,r] = geod_from_matrix(A)
    theta =angle(np.subtract(p,[c,0]))
    alpha = (theta)/2
    return [b,r*np.tan(alpha)]

def geod_from_points(p1, p2): #z,w are vectors
    # gives the geometric elements of the geodesic from two points: the starting point, end point, center of circle and radius
    p1 = np.array(p1, dtype='float64')
    p2 = np.array(p2, dtype='float64')
    if p1[0] == p2[0]:
        return [p1[0], INF]
    else:
        c = (p2[1]**2 - p1[1]**2 + p2[0]**2 - p1[0]**2)/(2 * (p2[0] - p1[0]))
        theta1 = angle(p1 - np.array([c, 0]))
        theta2 = angle(p2 - np.array([c, 0]))
        return [c, norm(p1-np.array([c, 0])), theta1, theta2]

def hyp_stop(p1, p2, perc):
    [c, r, theta1, theta2] = geod_from_points(p1, p2)
    return (theta2-theta1)*perc+theta1





class Actions(Scene):
    def construct(self):
        self.camera.background_color = WHITE  # "#ece6e2"
        reduction = [[1, 0, 0], [0, 1, 0]]
        ampl = [[1, 0], [0, 1], [0, 0]]
        num_col = 20
        lam_s, mu_s = val(2)
        a = 2
        b = 3
        c = 7
        y_max = 7
        y_min = -1
        theta = PI / sqrt(2)
        b = 0.1
        K = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        N = np.array([[1, b], [0, 1]])
        D = np.array([[1.1, 0], [0, 1 / 1.1]])
        G = np.array([[1, -2], [0, 1]]) @ np.array([[0, 1], [-1, 0]])
        A = np.array([[3, 2], [1, 1]])
        B = np.array([[1.48182, -1.14545], [0.190909, 0.527273]])
        plane = ComplexPlane(
            x_range=(-7 / SCALE, 7 / SCALE, 1 / SCALE),
            y_range=(y_min / SCALE, y_max / SCALE, 1 / SCALE),
            faded_line_ratio=2,
            # x_length=5,
            # y_length=2,
            axis_config={
                "unit_size": SCALE
            },
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 3,
                "stroke_opacity": 0.3
            },
        ).add_coordinates()
        # plane.set_color(WHITE)
        axis = Axes(
            x_range=[-7 / SCALE, 7 / SCALE, 1 / SCALE],
            y_range=[y_min / SCALE, y_max / SCALE, 1 / SCALE],
            x_length=20 / SCALE,
            y_length=(y_max - y_min) / SCALE,
            tips=False,
        )
        axis.set_color(BLACK)
        self.add(plane, axis)
        T = np.matrix([[1, 1], [0, 1]])
        J = np.matrix([[0, -1], [1, 0]])
        A_new = np.matrix([[5, -2/7], [1, 1/7]])
        # plane.n2p(vert[0])
        p_list = [[1,2], [1,4], [1,6]]
        #p_list = []
        for q in p_list:
            [c1,r1,c2,r2] = hyp_action_H(q,A_new)
            self.add(
                draw_geod(c2-r2, c2+r2, PINK, plane)
            )
            theta1 = 2*PI+angle(np.subtract([c2-r2,0],c1))
            theta2 = angle(np.subtract([c2 + r2, 0], c1))
            self.add(
                    Arc(r1, theta1, theta2 - theta1, arc_center=plane.n2p(D2_to_compl(c1)), stroke_width=3, color=PURE_RED)
            )
            dot = Dot(plane.n2p(D2_to_compl(q)), color=ORANGE)
            self.add(dot)


class Actions_parab(Scene):
    def construct(self):
        self.camera.background_color = WHITE  # "#ece6e2"
        reduction = [[1, 0, 0], [0, 1, 0]]
        ampl = [[1, 0], [0, 1], [0, 0]]
        num_col = 20
        lam_s, mu_s = val(2)
        a = 2
        b = 3
        c = 7
        y_max = 7
        y_min = -1
        theta = PI / sqrt(2)
        b = 0.1
        K = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        N = np.array([[1, b], [0, 1]])
        D = np.array([[1.1, 0], [0, 1 / 1.1]])
        G = np.array([[1, -2], [0, 1]]) @ np.array([[0, 1], [-1, 0]])
        A = np.array([[3, 2], [1, 1]])
        B = np.array([[1.48182, -1.14545], [0.190909, 0.527273]])
        plane = ComplexPlane(
            x_range=(-7 / SCALE, 7 / SCALE, 1 / SCALE),
            y_range=(y_min / SCALE, y_max / SCALE, 1 / SCALE),
            faded_line_ratio=2,
            # x_length=5,
            # y_length=2,
            axis_config={
                "unit_size": SCALE
            },
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 3,
                "stroke_opacity": 0.3
            },
        ).add_coordinates()
        # plane.set_color(WHITE)
        axis = Axes(
            x_range=[-7 / SCALE, 7 / SCALE, 1 / SCALE],
            y_range=[y_min / SCALE, y_max / SCALE, 1 / SCALE],
            x_length=20 / SCALE,
            y_length=(y_max - y_min) / SCALE,
            tips=False,
        )
        axis.set_color(BLACK)
        self.add(plane, axis)
        T = np.matrix([[1, 1], [0, 1]])
        J = np.matrix([[0, -1], [1, 0]])
        A_new = A @ N @ nl.inv(A)#np.matrix([[5, -2/7], [1, 1/7]])
        # plane.n2p(vert[0])
        p_list = [[1,1], [1,3.2], [1,5]]
        #p_list = []
        for q in p_list:
            [c1, r1, q_fixed] = parab_action_H(q,A_new)
            self.add(
                    Arc(r1, 0, 2*PI, arc_center=plane.n2p(D2_to_compl(c1)), stroke_width=3, color=PURE_GREEN)
            )
            dot = Dot(plane.n2p(D2_to_compl(q)), color=ORANGE)
            self.add(dot)
        dot = Dot(plane.n2p(D2_to_compl(q_fixed)), color=TEAL_D)
        self.add(dot)
        #### From here animation




class Actions_elliptic(Scene):
    def construct(self):
        self.camera.background_color = WHITE  # "#ece6e2"
        reduction = [[1, 0, 0], [0, 1, 0]]
        ampl = [[1, 0], [0, 1], [0, 0]]
        num_col = 20
        lam_s, mu_s = val(2)
        a = 2
        b = 3
        c = 7
        y_max = 7
        y_min = -1
        theta = PI / sqrt(2)
        b = 0.1
        K = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        N = np.array([[1, b], [0, 1]])
        D = np.array([[1.1, 0], [0, 1 / 1.1]])
        G = np.array([[1, -2], [0, 1]]) @ np.array([[0, 1], [-1, 0]])
        A = np.array([[3, 2], [1, 1]])
        B = np.array([[1.48182, -1.14545], [0.190909, 0.527273]])
        plane = ComplexPlane(
            x_range=(-7 / SCALE, 7 / SCALE, 1 / SCALE),
            y_range=(y_min / SCALE, y_max / SCALE, 1 / SCALE),
            faded_line_ratio=2,
            # x_length=5,
            # y_length=2,
            axis_config={
                "unit_size": SCALE
            },
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 3,
                "stroke_opacity": 0.3
            },
        ).add_coordinates()
        # plane.set_color(WHITE)
        axis = Axes(
            x_range=[-7 / SCALE, 7 / SCALE, 1 / SCALE],
            y_range=[y_min / SCALE, y_max / SCALE, 1 / SCALE],
            x_length=14 / SCALE,
            y_length=(y_max - y_min) / SCALE,
            tips=False,
        )
        axis.set_color(BLACK)
        self.add(plane, axis)
        T = np.matrix([[1, 1], [0, 1]])
        J = np.matrix([[0, -1], [1, 0]])
        #R = A @ K @ nl.inv(A)
        R = K
        q = fixpt(R)#+0.572*1j
        self.add(Dot(plane.n2p(q), color=PURE_BLUE))

        # plane.n2p(vert[0])
        p_list = [[1,2], [1,4], [1,6]]
        #p_list = []
        for q in p_list:
            [c, r] = ellip_action_H(q, R)
            #c = np.add(c,[0,0.572])
            theta1 = angle(np.subtract(q, c))
            self.add(
                    Arc(r, theta1, 2*PI, arc_center=plane.n2p(D2_to_compl(c)), stroke_width=3, color=PURE_BLUE)
            )
            dot = Dot(plane.n2p(D2_to_compl(q)), color=ORANGE)
            self.add(dot)


class Flows(Scene):
    def construct(self):
        self.camera.background_color = WHITE  # "#ece6e2"
        reduction = [[1, 0, 0], [0, 1, 0]]
        ampl = [[1, 0], [0, 1], [0, 0]]
        T = [0,0,0]
        num_col = 20
        velocity = 1
        p1 = np.array([0, 2])
        theta1 = PI / 8
        v1 = np.array([velocity * np.cos(theta1), velocity * np.sin(theta1)])
        lam_s, mu_s = val(2)
        a = 2
        b = 3
        c = 7
        y_max = 8
        y_min = -3
        theta = PI / sqrt(2)
        b = 0.1
        K = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        N = np.array([[1, b], [0, 1]])
        D = np.array([[1.1, 0], [0, 1 / 1.1]])
        G = np.array([[1, -2], [0, 1]]) @ np.array([[0, 1], [-1, 0]])
        A = np.array([[3, 2], [1, 1]])
        B = np.array([[1.48182, -1.14545], [0.190909, 0.527273]])
        plane = ComplexPlane(
            x_range=(-7 / SCALE, 7 / SCALE, 1 / SCALE),
            y_range=(y_min / SCALE, y_max / SCALE, 1 / SCALE),
            faded_line_ratio=2,
            # x_length=5,
            # y_length=2,
            axis_config={
                "unit_size": SCALE
            },
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 3,
                "stroke_opacity": 0.3
            },
        ).add_coordinates()
        # plane.set_color(WHITE)
        axis = Axes(
            x_range=[-7 / SCALE, 7 / SCALE, 1 / SCALE],
            y_range=[y_min / SCALE, y_max / SCALE, 1 / SCALE],
            x_length=20 / SCALE,
            y_length=(y_max - y_min) / SCALE,
            tips=False,
        )
        axis.set_color(BLACK)
        self.add(plane, axis)
        A = matrix_direction(p1,v1)
        [a,b,c,r] = geod_from_matrix(A)
        [a,h1] = uns_horo_flow(p1,v1)
        [b,h2] = stb_horo_flow(p1,v1)
        self.add(
            Circle(arc_center=plane.n2p(D2_to_compl([a,h1])), radius=h1, color=PURE_GREEN, stroke_width=3)
        )
        self.add(
            Circle(arc_center=plane.n2p(D2_to_compl([b,h2])), radius=h2, color=PURE_BLUE, stroke_width=3)
        )
        self.add(
            draw_geod(a,b,PURE_RED,plane)
        )
        # plane.n2p(vert[0])
        d1 = Dot(plane.n2p(D2_to_compl(p1)), color=ORANGE)

        num_start_balls = 5
        theta = angle(np.subtract(p1, [a, h1]))
        final_angle = angle([0,-h1])
        start_angles_ustb = linspace(theta, (final_angle-theta)*0.6+theta, num_start_balls)
        print("ust angles:", start_angles_ustb)
        theta = angle(np.subtract(p1, [b, h2]))
        final_angle = angle([0, h2])-2*PI
        start_angles_stb = linspace(theta, (final_angle - theta) * 0.3 + theta, num_start_balls)
        start_angles_flow = linspace(angle(np.subtract(p1, [c, 0])),
                                     hyp_stop(p1, [c+r, 0], 0.28),
                                     num_start_balls)
        centers_ustb = [np.multiply(h1, np.array([np.cos(t), np.sin(t)]))+np.array([a, h1]) for t in start_angles_ustb]
        centers_stb = [np.multiply(h2, np.array([np.cos(t), np.sin(t)])) + np.array([b, h2]) for t in start_angles_stb]
        centers_flow =[np.multiply(r, np.array([np.cos(t), np.sin(t)]))+np.array([c, 0]) for t in start_angles_flow]
        opty = linspace(0.7, 0.3, num_start_balls)
        starting_balls_ustb = VGroup(*[
            Dot(point=plane.n2p(D2_to_compl(centers_ustb[i])), color=PURE_GREEN, fill_opacity=opty[i],
                stroke_width=0)
            for i in range(num_start_balls)])
        starting_balls_stb = VGroup(*[
            Dot(point=plane.n2p(D2_to_compl(centers_stb[i])), color=PURE_BLUE, fill_opacity=opty[i],
                stroke_width=0)
            for i in range(num_start_balls)])
        starting_balls_flow = VGroup(*[
            Dot(point=plane.n2p(D2_to_compl(centers_flow[i])), color=PURE_RED, fill_opacity=opty[i],
                stroke_width=0)
            for i in range(num_start_balls)])
        self.add(starting_balls_ustb, starting_balls_stb, starting_balls_flow)
        d2 = Dot(plane.n2p(D2_to_compl([a,0])), color=ORANGE)
        d3 = Dot(plane.n2p(D2_to_compl([b,0])), color=ORANGE)
        label_z = MathTex("z", font_size=40, color=ORANGE).move_to(plane.n2p(0.3+p1[0]+p1[1] * 1j-0.12*1j))
        label_z1 = MathTex("z_{-\infty}", font_size=40, color=BLACK).move_to(plane.n2p(a-0.2*1j))
        label_z2 = MathTex("z_{\infty}", font_size=40, color=BLACK).move_to(plane.n2p(b -0.2 * 1j))
        self.add(d1,d2,d3,label_z,label_z1,label_z2)

        ### arrows
        u_scl = 1+0.4
        scl = 1 - 0.3
        angles_arrows = [PI/2+PI/3, PI/2, PI/2-PI/3.5]
        centers_arrows_ustb = [np.array([a, h1])+np.array([h1 * np.cos(t), h1* np.sin(t)]) for t in angles_arrows]
        end_arrows_ustb = [np.array([a, h1]) + np.array([h1 *u_scl * np.cos(t), h1* u_scl * np.sin(t)]) for t in angles_arrows]
        centers_arrows_stb = [np.array([b, h2]) + np.array([h2 * np.cos(t), h2 * np.sin(t)]) for t in angles_arrows]
        end_arrows_stb = [np.array([b, h2]) + np.array([h2 * scl * np.cos(t), h2*scl * np.sin(t)]) for t in angles_arrows]
        scl = 0.32
        centers_arrows_flow = [np.array([c, 0]) + np.array([r * np.cos(t), r * np.sin(t)]) for t in angles_arrows]
        end_arrows_flow = [np.array([c, 0])+np.array([r * np.cos(t), r * np.sin(t)]) + np.array([r * scl * np.cos(t-PI/2), r * scl * np.sin(t-PI/2)]) for t in angles_arrows]
        arrows_ustb = VGroup(*[
            Arrow(start=plane.n2p(D2_to_compl(centers_arrows_ustb[i])),
                  end=plane.n2p(D2_to_compl(end_arrows_ustb[i])),
                  buff=0,
                  stroke_width=2,
                  max_tip_length_to_length_ratio=0.4,
                  color=PURE_GREEN)
        for i in range(3)])
        arrows_stb = VGroup(*[
            Arrow(start=plane.n2p(D2_to_compl(centers_arrows_stb[i])),
                  end=plane.n2p(D2_to_compl(end_arrows_stb[i])),
                  buff=0,
                  stroke_width=2,
                  max_tip_length_to_length_ratio=0.2,
                  color=PURE_BLUE)
            for i in range(3)])
        arrows_flow = VGroup(*[
            Arrow(start=plane.n2p(D2_to_compl(centers_arrows_flow[i])),
                  end=plane.n2p(D2_to_compl(end_arrows_flow[i])),
                  buff=0,
                  stroke_width=2,
                  max_tip_length_to_length_ratio=0.22,
                  color=PURE_RED)
            for i in range(3)])
        self.add(arrows_ustb, arrows_stb, arrows_flow)






A = np.matrix([[2, 1], [0, 0]])
B = np.matrix([[2, 0], [0, 2]])
v = np.array([2, 3])
w = np.array([1, 1])
print(A @ B, v + w)
print(np.dot(A, B))
print(nl.inv(B))
