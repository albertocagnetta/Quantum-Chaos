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




def barnett_stadium(l1,l2,r1,r2):
    """
    gives the geometric paramteres for the barnett stadium, with two circles:
    center1 at [l1,0] with radius r1
    center2 at [0,l2] with radius r2
    :return: [x,y,theta1,theta2]
    :(x,y): intersection point of the two circles
    theta1 angle clockwise for the arc on the right
    theta2 angle counterclockwise for the arc on the top
    """
    # what returns?
    if l1**2+l2**2 > (r1+r2)**2 :
        print("too much distance between circles!")
        sys.exit("Error message")
    Q = - r2**2 + r1**2 + l2**2 - l1**2
    B = (l2**2)/(l1**2 + l2**2)*(l1/(l2**2)*Q - 2*l1)
    C = (l2**2)/(l1**2 + l2**2)*(Q**2/(4*l2**2) + l1**2 - r1**2)
    x = (-B-sqrt(B**2-4*C))/2
    y = l1/l2*x + Q/(2*l2)
    theta1 = acos((l1-x)/r1)
    theta2 = acos((l2-y)/r2)
    return [x, y, theta1, theta2]



def Barnett_billiard(p, v, num_col, x, y, center1, radius1, angle1, center2, radius2, angle2):
    point_list = [[p[0], p[1], 0]]
    for i in range(num_col):
        times = [line_intersection(p, v, np.array([0, -1, 0]), np.array([0, 0]), np.array([x, 0])),
                 line_intersection(p, v, np.array([-1, 0, 0]), np.array([0, 0]), np.array([0, y])),
                 circle_intersection(p, v, center1, radius1, PI-angle1, angle1, -1),
                 circle_intersection(p, v, center2, radius2, -PI / 2, angle2, -1)]
        hit_wall = times.index(min(times))
        print('times:', times)
        print("wall:", hit_wall, times[hit_wall])
        new_p = p + np.multiply(times[hit_wall], v)
        print("new position", new_p[0], new_p[1])
        if hit_wall == 0:
            v = bounce_line(v, np.array([0, -1]))
        elif hit_wall == 1:
            v = bounce_line(v, np.array([-1, 0]))
        elif hit_wall == 2:
            v = bounce_circle(v, new_p, center1)
        else:
            v = bounce_circle(v, new_p, center2)
        p = new_p
        print("nuova velocità", v)
        point_list.append([p[0], p[1], 0])
        print("fine turno\n")
    return point_list



class Draw_single(Scene):
    def construct(self):
        shift = np.array([-2,-2,0]) #vector to shift the image
        M = np.array([[1, 0], [0, 1], [0, 0]])
        "Variables for the Scene"
        velocity = 1  # velocity of the balls
        num_col = 10
        num_animation = 10
        radius = 0.05  # radius of the balls

        " Barnett stadium geometrical elements"
        l1 = 11
        l2 = 24
        r1 = 8
        r2 = 20
        center1 = np.array([l1,0])
        center2 = np.array([0,l2])
        [x, y, angle1, angle2] = barnett_stadium(l1, l2, r1, r2)
        boundary_color = BLUE
        boundary_width = 6
        #############################
        # the bx are the balls object for manim, optional
        p1 = np.array([1.5, 1])
        theta1 = PI / 3
        v1 = np.array([velocity * np.cos(theta1), velocity * np.sin(theta1)])
        b1 = Dot(np.add(shift,np.dot(M, p1)), radius=radius).set_color(YELLOW)
        render1 = Barnett_billiard(p1, v1, num_col, x, y, center1, r1, angle1, center2, r2, angle2)
        self.add(b1)
        # print(render1)

        # p2 = np.array([-1.5, 0])
        # theta2 = PI - PI / 3
        # v2 = np.array([velocity * np.cos(theta2), velocity * np.sin(theta2)])
        # b2 = Dot(np.add(shift,np.dot(M, p2)), radius=radius).set_color(PINK)
        # render2 = Barnett_billiard(p2, v2, n_col, x, y, center1, r1, angle1, center2, r2, angle2)
        # self.add(b2)
        # print(render2)

        # p3 = np.array([1,1.5])
        # theta3 = PI / 3
        # v3 = [velocity * np.cos(theta3), velocity * np.sin(theta3)]
        # b3 = Dot(np.add(shift,np.dot(M, p3)), radius=radius).set_color(BLUE)
        # render3 = Barnett_billiard(p3, v3, n_col, x, y, center1, r1, angle1, center2, r2, angle2)
        # self.add(b3)
        # print(render3)

        ###################################################
        " comment/uncomment these lines to leave traces behind balls "
        trace1 = TracedPath(b1.get_center, stroke_opacity=1, stroke_color=b1.get_color(), stroke_width=2)
        self.add(trace1)
        # trace2 = TracedPath(b2.get_center, stroke_opacity=[0,1], stroke_color=b2.get_color(), stroke_width=2, dissipating_time=0.5)
        # self.add(trace2)

        ###################################################
        " plotting boundary "
        self.add(Line(np.add([0, 0, 0], shift), np.add([l1 - r1, 0, 0], shift), stroke_width=boundary_width, color=boundary_color))
        self.add(Line(np.add([0, 0, 0], shift), np.add([0, l2 - r2, 0], shift), stroke_width=boundary_width, color=boundary_color))
        self.add(Arc(r1, PI, -angle1, arc_center=np.add([l1, 0, 0], shift), stroke_width=boundary_width, color=boundary_color))
        self.add(Arc(r2, -PI / 2, angle2, arc_center=np.add([0, l2, 0], shift), stroke_width=boundary_width, color=boundary_color))

        Anim1 = Succession(
            *[
                AnimationGroup(
                    # Create(Line(start=render1[i - 1], end=render1[i], stroke_width=2, color=b1.get_color())).set_run_time(
                    #    time(render1[i], render1[i - 1], 3 * velocity)),
                    MoveAlongPath(b1, Line(start=np.add(shift,render1[i - 1]),
                                           end=np.add(shift,render1[i])),
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
        #             MoveAlongPath(b2, Line(start=np.add(shift,render2[i - 1]),
    #                               end=np.add(shift,render2[i])), rate_func=linear).set_run_time(
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
        M = np.array([[1,0],[0,1],[0,0]])
        shift = np.array([-3.5,-3,0])
        ########################
        "parameters"
        velocity = 1 #velocity of the balls
        Num_points = 80 #number of bouncing balls
        num_col = 30
        num_animation = 30
        " Barnett stadium geometrical elements"
        l1 = 16.5
        l2 = 36
        r1 = 12
        r2 = 30
        center1 = np.array([l1, 0])
        center2 = np.array([0, l2])
        [x, y, angle1, angle2] = barnett_stadium(l1, l2, r1, r2)
        boundary_color = BLUE_E
        boundary_width = 5
        """
        the sample of the balls is distributed around point P with gaussian distribution of:
        - ray x_ray around x-coordinate of P;
        - ray y_ray around y-coordinate of P;
        another distribution is the linear one, you choose
        the points are saved in p_list 
        the starting angles are saved in theta_list, from start_theta to end_theta
        """
        P = np.array([1, 1.2])
        radius = 0.05  # radius of the balls
        Max_val_binom = 20
        x_ray = 0.2
        y_ray = 0.1
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
        end_theta = PI*(1/8)
        theta_list = linspace(start_theta, end_theta, Num_points)
        v_list = [[velocity*np.cos(a), velocity*np.sin(a)] for a in theta_list]
        #########################################################################

        "  The  billiard  simulation"
        List = []  # list of lists of points
        for l in range(Num_points):
            List.append(Barnett_billiard(p_list[l], v_list[l], num_col, x, y, center1, r1, angle1, center2, r2, angle2))

        # plotting the dots
        dots = VGroup(*[Dot(np.add(shift,np.dot(M, p_list[i])), radius=radius) for i in range(Num_points)])
        #dots.set_color_by_gradient(PINK, BLUE, YELLOW)  #color by gradient
        dots.set_color(PURE_GREEN)
        self.add(dots)
        ##############################
        " traces behind the balls, uncomment to use it, not to use with more then 50 balls "
        traces = VGroup(*[TracedPath(b.get_center, stroke_opacity=0.2, stroke_color=b.get_color(), stroke_width=2, dissipating_time=3.5) for b in dots])
        self.add(traces)
        ##############################
        " Plot the boundary "
        self.add(Line(np.add([0, 0, 0], shift), np.add([l1 - r1, 0, 0], shift), stroke_width=boundary_width,
                      color=boundary_color))
        self.add(Line(np.add([0, 0, 0], shift), np.add([0, l2 - r2, 0], shift), stroke_width=boundary_width,
                      color=boundary_color))
        self.add(Arc(r1, PI, -angle1, arc_center=np.add([l1, 0, 0], shift), stroke_width=boundary_width,
                     color=boundary_color))
        self.add(Arc(r2, -PI / 2, angle2, arc_center=np.add([0, l2, 0], shift), stroke_width=boundary_width,
                     color=boundary_color))

        TotalAnim = AnimationGroup(
            *[
                Succession(
                    *[
                        # the runtime option is needed to make balls move according to the space traveled
                        MoveAlongPath(dots[l], Line(start=np.add(shift,List[l][i - 1]),
                                                    end=np.add(shift,List[l][i])),
                                                    rate_func=linear).set_run_time(
                            time(List[l][i], List[l][i - 1], 3 * velocity))
                        for i in range(1, num_animation + 1)
                    ]
                )
                for l in range(Num_points)
            ]
        )
        self.play(TotalAnim)



