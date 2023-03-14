import numpy as np
from numpy import sign, dot, multiply, cos, sin, real, imag, matmul, subtract, linspace, float64
# from matplotlib.animation import FuncAnimation
import math
from math import sqrt, acos, tan
from mpmath import kleinj
# from sympy import Float
# from decimal import *
from manim import *
import string
import numpy.linalg as nl

from hyp_functions import *

# CONSTANTS
INF = float('inf')
EPS = 1e-12
EPSX = 1e-12

def hyp_billiard_bolza(p, v, centers, radii, num_col):
    """
    Simulate the movement of a point in a hyperbolic billiard (Bolza Surface)
    """
    C = 1 / np.sqrt(2) * np.array([[1j, 1], [1, 1j]])
    alpha = np.sqrt(np.sqrt(2) - 1)
    F = [np.array([[1 + np.sqrt(2), (2 + np.sqrt(2)) * alpha * np.exp(k * PI / 4 * 1j)],
                   [(2 + np.sqrt(2)) * alpha * np.exp(-k * PI / 4 * 1j), 1 + np.sqrt(2)]]) for k in range(4)]
    R = [nl.inv(C) @ A @ C for A in F]
    Rinv = [nl.inv(C) @ nl.inv(A) @ C for A in F]
    generators = [R[0],R[1],R[2],R[3],Rinv[0],Rinv[1],Rinv[2],Rinv[3]]
    A = matrix_direction(p, v)
    dir = 1
    new_p1 = p
    print("partenza e direzione:", new_p1, angle(v) * 180 / PI)
    new_v = v
    M = A  # the matrix that gives the sequence
    alphabet = list('A')
    letters = ['R0','R1','R2','R3','R4','R5','R6','R7']
    point_list = []
    geom_list = []
    point_list.append([p[0], p[1], 0])
    [c1, r1] = geod_from_point(p, v)
    print("matrice:", M)
    print("centro e raggio:", c1, r1)
    for i in range(num_col):
        print("volta", i + 1)
        # [a1,a2,c1,r1]=geod_from_matrix(M)
        geom_list.append([c1, r1])
        # x_inter = [inters_geodesic_and_line_wall(new_p1, new_v, SX_WALL)[0][0],
        #           inters_geodesic_and_line_wall(new_p1, new_v, DX_WALL)[0][0],
        #           inters_geodesic_and_circled_wall(new_p1, new_v, C1, R1)[0][0]]
        x_inter = [inters_geodesic_and_circled_wall(new_p1, new_v, centers[0], radii[0])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[1], radii[1])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[2], radii[2])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[3], radii[3])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[4], radii[4])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[5], radii[5])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[6], radii[6])[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, centers[7], radii[7])[0][0]]

        print("punti intersezione:", x_inter)
        if dir == 1:
            hit_wall = x_inter.index(min(filter(lambda a: a > new_p1[0] + EPSX, x_inter)))
            # hit_wall = times.index(min(x_inter))
        elif dir == -1:
            hit_wall = x_inter.index(max(filter(lambda a: a < new_p1[0] - EPSX, x_inter)))
            # hit_wall = times.index(min(x_inter))
        print('muro:', hit_wall)
        # print("nuova posizione", new_p[0], new_p[1])
        #for k in range(8):
        #    if hit_wall == k:
        k = hit_wall
        [new_p2, new_v] = inters_geodesic_and_circled_wall(new_p1, new_v, centers[k], radii[k])
        print("punto impatto:", new_p2)
        # new_v = inters_geodesic_and_circled_wall(new_p1, new_v, C1, R1)[1]
        point_list.append([new_p2[0], new_p2[1], 0])
        new_p1 = compl_to_2D(hyp_action(generators[(k)%8], D2_to_compl(new_p2)))
        new_v = hyp_action_vel(new_p2, new_v, generators[(k)%8])
        print("nuova partenza e direzione:", new_p1, angle(new_v) / PI * 180)
        point_list.append([new_p1[0], new_p1[1], 0])
        M = np.matmul(generators[(k)%8], M)
        print("matrice:", M)
        [c1, r1] = geod_from_point(new_p1, new_v)
        print("centro e raggio:", c1, r1)
        if (k)%8 != 2:
            dir = np.sign(c1-centers[(k+4)%8])
        else:
            dir = -np.sign(c1 - centers[6])
        print("direzione:", dir)
        alphabet = [letters[(k)%8]] + alphabet
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(alphabet)
    point_list.pop()
    print(len(point_list))
    print(len(geom_list))
    # geom_list.pop()
    return [point_list, geom_list]




def Bolza_data(C, points):
    # returns the data for the domain of Bolza surface on the upper-half-plane
    return [hyp_action(C, p) for p in points]




class Bolza_billiard(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        reduction = np.array([[1, 0, 0], [0, 1, 0]])
        ampl = np.array([[1, 0], [0, 1], [0, 0]])
        num_col = 10
        num_anim = 10
        C = 1 / np.sqrt(2) * np.array([[1j, 1], [1, 1j]])
        #self.camera.background_color = WHITE#"#ece6e2"

        h = -2.2

        # [p, v] = starting_from_M(A)
        p1 = [1, 2]
        #p0 = math.pow(2, -1 / 4) * np.exp(1j * (PI / 8 ))
        #p1 = compl_to_2D(hyp_action(nl.inv(C),p0-0.1))
        theta = -PI / 4
        velocity = 1
        v1 = [velocity * np.cos(theta), velocity * np.sin(theta)]
        b1 = Dot((np.dot(ampl, p1) + [0, h, 0]), radius=0.05).set_color(TEAL_D)  # (BLUE_C)

        # trace1 = TracedPath(b1.get_center, stroke_opacity=1, stroke_color=GREEN, stroke_width=2)
        self.add(b1)


        # geometric elements of the fundamental domain

        points = [math.pow(2, -1 / 4) * np.exp(1j * (PI / 8 + PI / 4 * k)) for k in range(8)]
        compl_vertices = Bolza_data(nl.inv(C), points)
        D2_vertices = [compl_to_2D(p) for p in Bolza_data(nl.inv(C), points)]
        centers = []
        radii = []
        angles = []
        for k in range(8):
            [c, r] = geod_H(compl_vertices[(k+3)%8], compl_vertices[(k+4)%8])
            [theta1, theta2] = geod_H_angles(compl_vertices[(k+3)%8], compl_vertices[(k+4)%8])
            centers.append(c)
            radii.append(r)
            angles.append([theta1, theta2])
        # the order of the side is the following :
        #           6
        #         7   5
        #        0      4
        #         1   3
        #           2
        #now we plot the domain
        colors = [PURE_GREEN, PURE_RED, PURE_BLUE, ORANGE, PURE_GREEN, PURE_RED, PURE_BLUE, ORANGE]
        for k in range(8):
            self.add(Arc(radii[k], angles[k][0], angles[k][1]-angles[k][0], arc_center=[centers[k], h, 0], color=colors[k]))


        # billiard:
        [point_list, geom_list] = hyp_billiard_bolza(p1, v1, centers, radii, num_col)

        # plot of the orbit
        list_anim = []
        for i in range(num_anim):
            q1 = point_list[2 * i]
            q2 = point_list[2 * i + 1]
            c = geom_list[i][0]
            r = geom_list[i][1]
            theta1 = angle(np.subtract(np.dot(reduction, q1), [c, 0]))
            theta2 = angle(np.subtract(np.dot(reduction, q2), [c, 0]))
            arc = Arc(r, theta1, theta2 - theta1, arc_center=[c, h, 0], stroke_width=2, color=BLUE)
            # .set_run_time(r*(theta2-theta1))
            group = AnimationGroup(
                MoveAlongPath(b1, arc, rate_func=smooth).set_run_time(1.5 * r * np.abs(theta2 - theta1)),
                Create(arc, rate_func=linear).set_run_time(1.5 * r * np.abs(theta2 - theta1))
            )
            list_anim.append(group)
            # self.play(MoveAlongPath(b1, Line([-0.5, h, 0], [-0.5, h, 0])))#Create(arc))
        # print(list_anim)
        Anim = Succession(*list_anim)
        self.play(Anim)  # rate_functions.ease_out_sine)



class Draw_multiple(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        """
        Class scene to draw multiples bouncing balls
        There is an option below to plot trace behind the balls, recommendable not to use with more than 50 balls
        """
        reduction = np.array([[1, 0, 0], [0, 1, 0]])
        ampl = np.array([[1, 0], [0, 1], [0, 0]])
        h = - 2.2
        shift = np.array([0,h,0])
        ########################
        "parameters"
        velocity = 1 #velocity of the balls
        speed = 2
        Num_points = 20 #number of bouncing balls
        num_col = 30
        #num_anim = 30
        " Barnett stadium geometrical elements"
        C = 1 / np.sqrt(2) * np.array([[1j, 1], [1, 1j]])

        boundary_width = 5
        """
        the sample of the balls is distributed around point P with gaussian distribution of:
        - ray x_ray around x-coordinate of P;
        - ray y_ray around y-coordinate of P;
        another distribution is the linear one, you choose
        the points are saved in p_list 
        the starting angles are saved in theta_list, from start_theta to end_theta
        """
        P = np.array([0, 1.2])
        radius = 0.05  # radius of the balls
        Max_val_binom = 20
        x_ray = 0.2
        y_ray = 0.2
        x_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        x_rnd = ((x_rnd) / Max_val_binom - 0.5) * x_ray / 0.5
        y_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        y_rnd = ((y_rnd) / Max_val_binom - 0.5) * y_ray / 0.5
        #print(((x_rnd)/Max_val_binom-0.5)*0.3/0.5)
        # linear distribution
        #x_rnd = [a for a in linspace(-x_ray, x_ray, Num_points)]
        #y_rnd = [a for a in linspace(-y_ray, y_ray, Num_points)]
        p_list = [np.add(P, [x_rnd[i], y_rnd[i]]) for i in range(Num_points)]

        start_theta = PI*0
        end_theta = PI*(1/6)
        theta_list = linspace(start_theta, end_theta, Num_points)
        v_list = [[velocity*np.cos(a), velocity*np.sin(a)] for a in theta_list]

        # plotting the dots
        dots = VGroup(*[Dot(np.add(shift, np.dot(ampl, p_list[i])), radius=radius) for i in range(Num_points)])
        dots.set_color_by_gradient(BLUE_C, TEAL_C, ORANGE)  # dots.set_color_by_gradient(PINK, BLUE, YELLOW)  #color by gradient
        # dots.set_color(PURE_GREEN)
        self.add(dots)

        #########################################################################
        " Plot the boundary "
        points = [math.pow(2, -1 / 4) * np.exp(1j * (PI / 8 + PI / 4 * k)) for k in range(8)]
        compl_vertices = Bolza_data(nl.inv(C), points)
        D2_vertices = [compl_to_2D(p) for p in Bolza_data(nl.inv(C), points)]
        centers = []
        radii = []
        angles = []
        for k in range(8):
            [c, r] = geod_H(compl_vertices[(k + 3) % 8], compl_vertices[(k + 4) % 8])
            [theta1, theta2] = geod_H_angles(compl_vertices[(k + 3) % 8], compl_vertices[(k + 4) % 8])
            centers.append(c)
            radii.append(r)
            angles.append([theta1, theta2])
        # the order of the side is the following :
        #           6
        #         7   5
        #        0      4
        #         1   3
        #           2
        # now we plot the domain
        colors = [PURE_GREEN, PURE_RED, PURE_BLUE, ORANGE, PURE_GREEN, PURE_RED, PURE_BLUE, ORANGE]
        for k in range(8):
            self.add(Arc(radii[k], angles[k][0], angles[k][1] - angles[k][0], arc_center=[centers[k], h, 0],
                         color=colors[k], stroke_width=boundary_width))

        ##############################################################
        "  The  billiard  simulation"
        List = []  # list of lists of points
        for l in range(Num_points):
            List.append(
                hyp_billiard_bolza(p_list[l], v_list[l], centers, radii, num_col)
            )


        ##############################
        " traces behind the balls, uncomment to use it, not to use with more then 50 balls "
        #traces = VGroup(*[TracedPath(b.get_center, stroke_opacity=0.2, stroke_color=b.get_color(), stroke_width=2, dissipating_time=3.5) for b in dots])
        #self.add(traces)
        # List_anim = []
        # for k in range(Num_points):
        #     list_anim = []
        #     for i in range(num_col):
        #         q1 = List[k][0][2*i]#point_list[2 * i]
        #         q2 = List[k][0][2*i+1]#point_list[2 * i + 1]
        #         c = List[k][1][i][0]#geom_list[i][0]
        #         r = List[k][1][i][1]#geom_list[i][1]
        #         theta1 = angle(np.subtract(np.dot(reduction, q1), [c, 0]))
        #         theta2 = angle(np.subtract(np.dot(reduction, q2), [c, 0]))
        #         arc = Arc(r, theta1, theta2 - theta1, arc_center=[c, h, 0], stroke_width=2, color=BLUE)
        #         # .set_run_time(r*(theta2-theta1))
        #         group = AnimationGroup(
        #             MoveAlongPath(dots[k], arc, rate_func=smooth).set_run_time(speed * r * np.abs(theta2 - theta1)),
        #             Create(arc, rate_func=linear, opacity=0.4).set_run_time(speed * r * np.abs(theta2 - theta1))
        #         )
        #         list_anim.append(group)
        #         # self.play(MoveAlongPath(b1, Line([-0.5, h, 0], [-0.5, h, 0])))#Create(arc))
        #     # print(list_anim)
        #     List_anim.append(Succession(*list_anim))
        #
        # TotalAnim = AnimationGroup(*List_anim)
        # self.play(TotalAnim)






