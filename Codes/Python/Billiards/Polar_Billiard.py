import pygame
import numpy as np
from manim import *
from numpy import sin, cos, pi, linspace, sign, dot, multiply, add, roots
import sympy as sp
from sympy import cos, sin, Symbol, diff, div, Abs, sqrt

from functions import *
#init_printing(use_unicode=False, wrap_line=False)
import math

PI = math.pi
INF = 1000
EPS = 1e-12
t = Symbol('t') #setting the variable
x = Symbol('x')
y = Symbol('y')


def polar_billiard(p, v, polar, n_col, boundary):
    point_list=[[p[0],p[1],0]]
    #boundary = Matrix([polar*sp.cos(t), polar*sp.sin(t)])
    for i in range(n_col):
        #time = polar_wall_intersection(p, v, polar)
        time = polar_wall_intersection(p, v, boundary)
        #print("tempi impatto", time)#, time2)
        p = p+np.multiply(time, v)
        #print("nuova posizione", new_p[0], new_p[1])
        v = bounce_polar(v, p, polar)
        #p = new_p
        #print("rimbalzo, numero volta", p, i+1)
        # converting float sympy array to numpy float array, because manim doesn't know hw to handle sympy
        point = np.array(p, dtype=np.float64)
        point_list.append([point[0], point[1], 0])
    return point_list







class Draw_single(Scene):
    def construct(self):
        M = np.array([[1, 0], [0, 1], [0, 0]])
        shift = np.array([1.2, 0, 0])  # vector to shift the image
        """
        Choose your boundary, it is sufficient to uncomment one of the following sections
        """
        boundary_color = BLUE_E
        boundary_width = 5
        ######### cardioid
        r = 1.2
        polar = 2 * r * (1 - sp.cos(t))
        boundary = (x ** 2 + y ** 2) ** 2 + 4 * r * x * (x ** 2 + y ** 2) - 4 * r ** 2 * y ** 2
        ######## ellipse
        # a, b = 3,1.2 (b<a)
        # e = np.sqrt(1-(b/a)**2)
        # polar = b/sp.sqrt(1- (e*sp.cos(t))**2)
        # boundary = (x**2)/a**2+(y**2)/b**2=1
        ######## Cassini oval
        # b, e = 4, 1.1
        # B = -(b / e) ** 2 * sp.cos(2 * t)
        # C = -(b ** 4 - (b / e) ** 4)
        ## polar equation r^4+2Br^2+C=0
        # polar = sp.sqrt(-B + sp.sqrt(B ** 2 - C))
        # boundary = (x ** 2 + y ** 2) ** 2 - 2 * (b / e) ** 2 * (x ** 2 - y ** 2) + (b / e) ** 4 - b ** 4
        ######### Sphere induced by the 2k-norm (k positive integer)
        # k, r = 2, 3 #(2k)-norm, (2k)-ball o radius r
        # polar = r/(sp.cos(t)**(2*k)+sp.sin(t)**(2*k))**(1/(2*k))#r/(sp.Abs(sp.cos(t))**(2*k)+sp.Abs(sp.sin(t))**(2*k))**(1/(2*k))
        # boundary = x**(2*k)+y**(2*k)-r**(2*k)

        "Variables for the Scene"
        num_col = 20  # number of collisions
        num_animation = 5 # number of animation to display
        velocity = 1
        radius = 0.05 #radius of the balls

        ###################### balls to display, from 1 to 3, uncomment or comment to hide/show them
        # the b are the ball object for manim, optional
        p1 = np.array([0, 0])
        theta1 = PI / 4
        v1 = [velocity * np.cos(theta1), velocity * np.sin(theta1)]
        b1 = Dot(shift+np.dot(M, p1), radius=radius).set_color(YELLOW)
        self.add(b1)
        render1 = polar_billiard(p1, v1, polar, num_col, boundary)

        p2 = [-1, 0]
        theta2 = PI / 8
        v2 = [velocity * np.cos(theta2), velocity * np.sin(theta2)]
        b2 = Dot(shift+np.dot(M, p2), radius=radius).set_color(PINK)
        self.add(b2)
        render2 = polar_billiard(p2, v2, polar, num_col, boundary)

        # p3 = [1,1.5]
        # theta3 = PI / 3
        # v3 = [velocity * np.cos(theta3), velocity * np.sin(theta3)]
        # b3 = Dot(shift+np.dot(M, p3), radius=radius).set_color(BLUE)
        # self.add(b3)
        # render2 = polar_billiard(p3, v3, polar, num_col, boundary)

        ###################################################
        " comment/uncomment these lines to leave traces behind balls "
        trace1 = TracedPath(b1.get_center, stroke_opacity=1, stroke_color=b1.get_color(), stroke_width=2)
        self.add(trace1)
        trace2 = TracedPath(b2.get_center, stroke_opacity=1, stroke_color=b2.get_color(), stroke_width=2)
        self.add(trace2)
        # trace3 = TracedPath(b3.get_center, stroke_opacity=1, stroke_color=b3.get_color(), stroke_width=2)
        # self.add(trace3)

        ###################################################
        " Plot the boundary"
        # radial bound for Cardioid  2 * r * (1 - np.cos(u))
        # radial bound for ellipse b/sp.sqrt(1- (e*np.cos(u))**2)
        # radial bound for 2k-norm:  r/(np.cos(u)**q+np.sin(u)**q)**(1/q)
        # radial bound for Cassini oval: np.sqrt(-(-(b/e)**2*np.cos(2*u))+np.sqrt((-(b/e)**2*np.cos(2*u))**2-C))
        boundary = ParametricFunction(
            lambda u: np.array([
                shift[0] + 2 * r * (1 - np.cos(u))*np.cos(u),
                shift[1] + 2 * r * (1 - np.cos(u))*np.sin(u),
                0
            ]), stroke_width=boundary_width, color=boundary_color, t_range=np.array([0, 2 * PI]), fill_opacity=0
        )
        self.add(boundary)
        Anim1 = Succession(
            *[
                AnimationGroup(
                    MoveAlongPath(b1, Line(start=np.add(shift,render1[i - 1]),
                                           end=np.add(shift, render1[i])),
                                  rate_func=linear).set_run_time(time(render1[i - 1], render1[i], 3 * velocity))
                )
                for i in range(1, num_animation + 1)
            ]
        )
        Anim2 = Succession(
            *[
                AnimationGroup(
                    MoveAlongPath(b2, Line(start=np.add(shift, render2[i - 1]),
                                           end=np.add(shift, render2[i])),
                                  rate_func=linear).set_run_time(time(render2[i - 1], render2[i], 3 * velocity))
                )
                for i in range(1, num_animation + 1)
            ]
        )
        # Anim3 = Succession(
        #     *[
        #         AnimationGroup(
        #             MoveAlongPath(b3, Line(start=np.add(shift, render3[i - 1]),
        #                                    end=np.add(shift, render3[i])),
        #                           rate_func=linear).set_run_time(time(render3[i - 1], render3[i], 3 * velocity))
        #         )
        #         for i in range(1, num_animation + 1)
        #     ]
        # )
        self.play(Anim1, Anim2)#,Anim3)



class Draw_multiple(Scene):
    def construct(self):
        """
        Class scene to draw multiples bouncing balls
        There is an option below to plot trace behind the balls, recommendable not to use with more than 50 balls
        """
        M = np.array([[1,0],[0,1],[0,0]])
        shift = np.array([1.2,0,0]) # vector to shift the image
        """
        Choose your boundary, it is sufficient to uncomment one of the following sections
        """
        boundary_color = BLUE_E
        boundary_width = 5
        ######### cardioid
        r = 1.8
        polar = 2 * r * (1 - sp.cos(t))
        boundary = (x ** 2 + y ** 2) ** 2 + 4 * r * x * (x ** 2 + y ** 2) - 4 * r ** 2 * y ** 2
        ######## ellipse
        # a, b = 3,1.2 (b<a)
        # e = np.sqrt(1-(b/a)**2)
        # polar = b/sp.sqrt(1- (e*sp.cos(t))**2)
        # boundary = (x**2)/a**2+(y**2)/b**2=1
        ######## Cassini oval
        # b, e = 4, 1.1
        # B = -(b / e) ** 2 * sp.cos(2 * t)
        # C = -(b ** 4 - (b / e) ** 4)
        ## polar equation r^4+2Br^2+C=0
        # polar = sp.sqrt(-B + sp.sqrt(B ** 2 - C))
        # boundary = (x ** 2 + y ** 2) ** 2 - 2 * (b / e) ** 2 * (x ** 2 - y ** 2) + (b / e) ** 4 - b ** 4
        ######### Sphere induced by the 2k-norm (k positive integer)
        # k, r = 2, 3 #(2k)-norm, (2k)-ball o radius r
        # polar = r/(sp.cos(t)**(2*k)+sp.sin(t)**(2*k))**(1/(2*k))#r/(sp.Abs(sp.cos(t))**(2*k)+sp.Abs(sp.sin(t))**(2*k))**(1/(2*k))
        # boundary = x**(2*k)+y**(2*k)-r**(2*k)

        ########################
        "parameters"
        velocity = 1 #velocity of the balls
        Num_points = 10 #number of bouncing balls
        num_col = 6
        num_animation = 6
        """
        the sample of the balls is distributed around point P with gaussian distribution of:
        - ray x_ray around x-coordinate of P;
        - ray y_ray around y-coordinate of P;
        another distribution is the linear one, you choose
        the points are saved in p_list 
        the starting angles are saved in theta_list, from start_theta to end_theta
        """
        P = np.array([-1,1])
        radius = 0.1  # radius of the balls
        Max_val_binom = 20
        x_ray = 0.3
        y_ray = 0.15
        #x_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        #x_rnd = ((x_rnd) / Max_val_binom - 0.5) * x_ray / 0.5
        #y_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        #y_rnd = ((y_rnd) / Max_val_binom - 0.5) * y_ray / 0.5
        #print(((x_rnd)/Max_val_binom-0.5)*0.3/0.5)
        # linear distribution
        x_rnd = [a for a in linspace(-x_ray, x_ray, Num_points)]
        y_rnd = [a for a in linspace(-y_ray, y_ray, Num_points)]
        p_list = [np.add(P, [x_rnd[i], y_rnd[i]]) for i in range(Num_points)]

        start_theta = PI*0
        end_theta = PI*0
        theta_list = linspace(start_theta, end_theta, Num_points)
        v_list = [[velocity*np.cos(a), velocity*np.sin(a)] for a in theta_list]
        #########################################################################

        "  The  billiard  simulation"
        List = []  # list of lists of points
        for l in range(Num_points):
            List.append(polar_billiard(p_list[l], v_list[l], polar, num_col, boundary))

        # plotting the dots
        dots = VGroup(*[Dot(np.add(np.dot(M, p_list[i]),shift), radius=radius) for i in range(Num_points)])
        dots.set_color_by_gradient(PINK, BLUE, YELLOW)  #color by gradient
        #dots.set_color(BLUE)
        self.add(dots)
        ##############################
        " traces behind the balls, uncomment to use it, not to use with more then 50 balls "
        traces = VGroup(*[TracedPath(b.get_center, stroke_opacity=0.8, stroke_color=b.get_color(), stroke_width=2, dissipating_time=5) for b in dots])
        self.add(traces)
        ##############################
        ###################################################
        " Plot the boundary"
        # radial bound for Cardioid  2 * r * (1 - np.cos(u))
        # radial bound for ellipse b/sp.sqrt(1- (e*np.cos(u))**2)
        # radial bound for 2k-norm:  r/(np.cos(u)**q+np.sin(u)**q)**(1/q)
        # radial bound for Cassini oval: np.sqrt(-(-(b/e)**2*np.cos(2*u))+np.sqrt((-(b/e)**2*np.cos(2*u))**2-C))
        boundary = ParametricFunction(
            lambda u: np.array([
                shift[0] + 2 * r * (1 - np.cos(u)) * np.cos(u),
                shift[1] + 2 * r * (1 - np.cos(u)) * np.sin(u),
                0
            ]), stroke_width=boundary_width, color=boundary_color, t_range=np.array([0, 2 * PI]), fill_opacity=0
        )
        self.add(boundary)

        TotalAnim = AnimationGroup(
            *[
                Succession(
                    *[
                        # the runtime option is needed to make balls move according to the space traveled
                        MoveAlongPath(dots[l], Line(start=np.add(List[l][i-1],shift),
                                                    end=np.add(List[l][i],shift)),
                                      rate_func=linear).set_run_time(time(List[l][i], List[l][i - 1], 3 * velocity))
                        for i in range(1, num_animation + 1)
                    ]
                )
                for l in range(Num_points)
            ]
        )
        self.play(TotalAnim)
