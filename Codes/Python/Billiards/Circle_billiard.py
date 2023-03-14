import numpy as np
from manim import *
import sympy as sp
from sympy import cos, sin, Symbol, diff, div, Abs, sqrt
from math import sqrt, acos, tan
from scipy import sparse
from scipy.sparse import linalg as sla
from numpy import sin, cos, pi, linspace, sign, dot, multiply, add, roots
import numpy.linalg as nl

from functions import *

INF = 1000
EPS = 1e-12


def Circle_billiard(p, v, n_col, center, R):
    point_list = [[p[0], p[1], 0]]
    for i in range(n_col):
        times = [circle_intersection(p, v, center, R, -PI, 2*PI, 1)]
        hit_wall = times.index(min(times))
        print('times:', times)
        print("wall:", hit_wall, times[hit_wall])
        new_p = p + np.multiply(times[hit_wall], v)
        print("new position", new_p[0], new_p[1])
        if hit_wall == 0:
            v = bounce_circle(v, new_p, center)
        p = new_p
        print("nuova velocità", v)
        point_list.append([p[0], p[1], 0])
        print("fine turno\n")
    return point_list


class Draw_single(Scene):
    def construct(self):
        M = np.array([[1, 0], [0, 1], [0, 0]])
        "Variables for the Scene"
        velocity = 1  # velocity of the balls
        num_col = 6
        num_animation = 6
        radius = 0.05  # radius of the balls

        " Circle geometrical elements"
        Radius = 2.2
        center = np.array([0, 0])
        boundary_color = BLUE
        boundary_width = 5
        #############################
        # the bx are the balls object for manim, optional
        p1 = np.array([1.5, 0])
        theta1 = PI / 3
        v1 = np.array([velocity * np.cos(theta1), velocity * np.sin(theta1)])
        b1 = Dot(np.dot(M, p1), radius=radius).set_color(YELLOW)
        render1 = Circle_billiard(p1, v1, num_col, center, Radius)
        self.add(b1)
        # print(render1)

        # p2 = np.array([-1.5, 0])
        # theta2 = PI - PI / 3
        # v2 = np.array([velocity * np.cos(theta2), velocity * np.sin(theta2)])
        # b2 = Dot(np.dot(M, p2), radius=radius).set_color(PINK)
        # render2 = Circle_billiard(p2, v2, num_col, L, Radius)
        # self.add(b2)
        # print(render2)

        # p3 = np.array([1,1.5])
        # theta3 = PI / 3
        # v3 = [velocity * np.cos(theta3), velocity * np.sin(theta3)]
        # b3 = Dot(np.dot(M, p3), radius=radius).set_color(BLUE)
        # render3 = Circle_billiard(p3, v3, num_col, L, Radius)
        # self.add(b3)
        # print(render3)

        ###################################################
        " comment/uncomment these lines to leave traces behind balls "
        trace1 = TracedPath(b1.get_center, stroke_opacity=1, stroke_color=b1.get_color(), stroke_width=2)
        self.add(trace1)
        # trace2 = TracedPath(b2.get_center, stroke_opacity=[0,1], stroke_color=b2.get_color(), stroke_width=2, dissipating_time=0.5)
        # self.add(trace2)
        #trace3 = TracedPath(b3.get_center, stroke_opacity=1, stroke_color=b3.get_color(), stroke_width=2)
        #self.add(trace3)

        ###################################################
        " plotting boundary "
        self.add(Arc(Radius, PI / 2, 2*PI, arc_center=np.dot(M,center), stroke_width=boundary_width, color=boundary_color))

        Anim1 = Succession(
            *[
                AnimationGroup(
                    # Create(Line(start=render1[i - 1], end=render1[i], stroke_width=2, color=b1.get_color())).set_run_time(
                    #    time(render1[i], render1[i - 1], 3 * velocity)),
                    MoveAlongPath(b1, Line(start=render1[i - 1], end=render1[i]),
                                  rate_func=linear).set_run_time(
                        time(render1[i - 1], render1[i], 3 * velocity))
                )
                for i in range(1, num_animation + 1)
            ]
        )
        # Anim2 = Succession(
        #     *[
        #         AnimationGroup(
        #             #Create(Line(start=render2[i - 1], end=render2[i], stroke_width=2).set_color(PINK)).set_run_time(
        #             #    time(render2[i], render2[i - 1], 3 * velocity)),
        #             MoveAlongPath(b2, Line(start=render2[i - 1], end=render2[i]), rate_func=linear).set_run_time(
        #                 time(render2[i - 1], render2[i], 3 * velocity))
        #         )
        #         for i in range(1, num_animation + 1)
        #     ]
        # )
        for i in range(1, num_animation + 1):
            print(render1[i - 1], render1[i])
        self.play(Anim1)  # , Anim2)


class Draw_multiple(Scene):
    def construct(self):
        """
        Class scene to draw multiples bouncing balls
        There is an option below to plot trace behind the balls, recommendable not to use with more than 50 balls
        """
        M = np.array([[1, 0], [0, 1], [0, 0]])

        ########################
        "parameters"
        velocity = 1  # velocity of the balls
        Num_points = 10  # number of bouncing balls
        num_col = 6
        num_animation = 6
        " Circle geometrical elements"
        center = np.array([0, 0])
        Radius = 1.7
        boundary_color = BLUE
        boundary_width = 5
        """
        the sample of the balls is distributed around point P with gaussian distribution of:
        - ray x_ray around x-coordinate of P;
        - ray y_ray around y-coordinate of P;
        another distribution is the linear one, you choose
        the points are saved in p_list 
        the starting angles are saved in theta_list, from start_theta to end_theta
        """
        P = np.array([-0.5, 0.8])
        radius = 0.05  # radius of the balls
        Max_val_binom = 20
        x_ray = 0.3
        y_ray = 0.15
        x_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        x_rnd = ((x_rnd) / Max_val_binom - 0.5) * x_ray / 0.5
        y_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        y_rnd = ((y_rnd) / Max_val_binom - 0.5) * y_ray / 0.5

        # linear distribution
        #x_rnd = [a for a in linspace(-x_ray, x_ray, Num_points)]
        #y_rnd = [a for a in linspace(-y_ray, y_ray, Num_points)]
        p_list = [np.add(P, [x_rnd[i], y_rnd[i]]) for i in range(Num_points)]

        start_theta = PI * 0
        end_theta = PI * 0
        theta_list = linspace(start_theta, end_theta, Num_points)
        v_list = [[velocity * np.cos(a), velocity * np.sin(a)] for a in theta_list]
        #########################################################################

        "  The  billiard  simulation"
        List = []  # list of lists of points
        for l in range(Num_points):
            List.append(Circle_billiard(p_list[l], v_list[l], num_col, center, Radius))

        # plotting the dots
        dots = VGroup(*[Dot(np.dot(M, p_list[i]), radius=radius) for i in range(Num_points)])
        dots.set_color_by_gradient(PINK, BLUE, YELLOW)  # color by gradient
        # dots.set_color(BLUE)
        self.add(dots)
        ##############################
        " traces behind the balls, uncomment to use it, not to use with more then 50 balls "
        traces = VGroup(*[
            TracedPath(b.get_center, stroke_opacity=0.8, stroke_color=b.get_color(), stroke_width=2, dissipating_time=5)
            for b in dots])
        self.add(traces)
        ##############################
        " Plot the boundary "
        self.add(Arc(Radius, PI / 2, 2*PI, arc_center=np.dot(M,center), stroke_width=boundary_width, color=boundary_color))

        TotalAnim = AnimationGroup(
            *[
                Succession(
                    *[
                        # the runtime option is needed to make balls move according to the space traveled
                        MoveAlongPath(dots[l], Line(start=List[l][i - 1], end=List[l][i]),
                                      rate_func=linear).set_run_time(
                            time(List[l][i], List[l][i - 1], 3 * velocity))
                        for i in range(1, num_animation + 1)
                    ]
                )
                for l in range(Num_points)
            ]
        )
        self.play(TotalAnim)



