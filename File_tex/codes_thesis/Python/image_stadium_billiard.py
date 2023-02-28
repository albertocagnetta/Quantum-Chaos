import numpy as np
from manim import *

from intersect import intersection
import matplotlib.pyplot as plt
from math import sqrt, acos, tan
from scipy import sparse
from scipy.sparse import linalg as sla
from numpy import sin, cos, pi, linspace, abs, sign, subtract, random

"""
:type direction: [vx,vy] directional vector
:type b_point: bouncing point
:type center: center of the circle
"""

#costants
#SIZEX=3000
#SIZEY=1000


BKground=(196,203,211)
INF = 1000
EPS = 1e-11

def getTragectory(p,v): # p is the position, v is the velocity vector
    #xc=x+y*m
    return [p,v] #[xc,0,math.sqrt((xc-x)**2+y**2)]

def bounce_circle(direction, b_point, center):
    """
    Calculate the new direction after a bouncing with a circled wall
    """
    F1 = [[b_point[0]-center[0], b_point[1]-center[1]], [b_point[1]-center[1], center[0]-b_point[0]]]
    F2 = [[-1, 0], [0, 1]]
    F3 = [[center[0]-b_point[0],center[1]-b_point[1]],[-(b_point[1]-center[1]),b_point[0]-center[0]]]
    #F3 = [[b_point[0]-center[0],b_point[1]-center[1]],[-b_point[1]+center[1],-b_point[0]+center[0]]]
    F3 = -np.dot(1/((b_point[0]-center[0])**2+(b_point[1]-center[1])**2), F3)
    return np.dot(np.matmul(np.matmul(F1, F2), F3), direction)

#print("prova:", np.dot([[0,1],[-1,0]],[2,3]))
#print("nuova direzione", bounce_circle([1,1],[np.sqrt(Radius)+WIDTH/2,np.sqrt(Radius)],[WIDTH/2,0]))


def upper_wall_intersection(p,v,Radius):
    l, r = 0, INF
    while r-l > EPS:
        m = (l+r)/2
        if p[1]+m*v[1] > Radius:
            r = m
        else:
            l = m
    return l


def lower_wall_intersection(p,v,Radius):
    l, r = 0, INF
    while r-l > EPS:
        m = (l+r)/2
        if p[1]+m*v[1] < -Radius:
            r = m
        else:
            l = m
    return l


def sgn(x):
    return int(x > 0) - int(x < 0)
def norm(v):
    return np.sqrt(np.dot(v, v))


def right_c_intersection(p, v, L, Radius):
    c = [L / 2, 0]
    #if v[0] >= 0:
    #l, r = 0, (2 * (Radius) + L)
    test_p = p + np.multiply(np.dot(np.subtract(c, p), v), v)
    #print("Norma prova: ", norm(test_p-c))
    #print("prodotto direzione prova: ", np.dot(np.subtract(c, p), v))
    #print("Punto prova: ",test_p)
    if norm(test_p-c) > Radius:
        return INF
    elif np.dot(np.subtract(c, p), v) < 0:
        return INF
    else:
        t = np.dot(np.subtract(c, p), v)+np.sin(acos(norm(test_p-c)/Radius))*Radius
        test_p = p + np.multiply(t, v)
        if test_p[0] > c[0]:
            return t
        else:
            return INF
        #l, r = np.dot(np.subtract(c, p), v), 2*Radius
        #while r - l > EPS:
        #    m = (l + r) / 2
        #    new_p = p + np.multiply(m, v)
        #    if new_p[0] > c[0]:
        #        r = m
        #    else:
        #        l = m
        #return l


def left_c_intersection(p, v, L, Radius):
    c = [-L / 2, 0]
    #if v[0] >= 0:
    #l, r = 0, (2 * (Radius) + L)
    test_p = p + np.multiply(np.dot(np.subtract(c, p), v), v)
    #print("Norma prova: ", norm(test_p-c))
    #print("prodotto direzione prova: ", np.dot(np.subtract(c, p), v))
    #print("Punto prova: ",test_p)
    if norm(test_p-c) > Radius:
        return INF
    elif np.dot(np.subtract(c, p), v) < 0:
        return INF
    else:
        t = np.dot(np.subtract(c, p), v)+np.sin(acos(norm(test_p-c)/Radius))*Radius
        test_p = p + np.multiply(t, v)
        if test_p[0] < c[0]:
            return t
        else:
            return INF




def billiard(p, v, n_col, L, Radius):
    point_list=[[p[0], p[1], 0]]
    for i in range(n_col):
        times = [upper_wall_intersection(p, v, Radius), lower_wall_intersection(p, v, Radius),
                 left_c_intersection(p, v, L, Radius), right_c_intersection(p, v, L, Radius)]
        hit_wall = times.index(min(times))
        #print('tempi:', times)
        #print("muro:", hit_wall, times[hit_wall])
        new_p = p+np.multiply(times[hit_wall], v)
        #print("nuova posizione", new_p[0], new_p[1])
        if hit_wall == 0 or hit_wall == 1:
            v[1] = -v[1]
        elif hit_wall == 2:
            v = bounce_circle(v, new_p, [-L/2, 0])
        else:
            v = bounce_circle(v, new_p, [L/2, 0])
        p = new_p
        point_list.append([p[0], p[1], 0])
    return point_list


def time(p1, p2, velocity):  #computes the distance done and the time of the path between two points
    #sequence = billiard(p, v, num_col, L, Radius)
    length = 0
    #for i in range(1,num_col+1):
    #    length = length + norm(np.subtract(p1, p2))
    #return length/norm(v)
    return norm(np.subtract(p1, p2))/velocity


def stop(p1, p2, perc):
    p1 = np.array(p1,dtype='float64')
    p2 = np.array(p2,dtype='float64')
    return np.array([[1,0],[0,1],[0,0]],dtype='float64') @ ((1-perc)*p1+perc*p2)




class Draw_single(Scene):
    def construct(self):
        self.camera.background_color = WHITE#"#ece6e2"

        stadium_back = WHITE#"#c2c4c6"#f2f4f6"#"#8598a7"#"#069fd9"#"#05014a"
        stadium_bord = "#353c42"
        col_orb1 = "#0000ff"#"#05014a"

        M = [[1, 0], [0, 1], [0, 0]]
        reduc = [[1,0,0],[0,1,0]]
        velocity = 1
        # the b are the ball object for manim, optional
        p1 = [0, 0]
        theta1 = PI / 4
        v1 = [velocity * np.cos(theta1), velocity * np.sin(theta1)]
        r_ball1 = 0.08
        b1_start = Dot(np.dot(M, p1), radius=r_ball1).set_color(col_orb1)
        b1 = Dot(np.dot(M, p1), radius=r_ball1).set_color(col_orb1)

        # p2 = [-1.5, 0]
        # theta2 = PI-PI / 3
        # v2 = [velocity * np.cos(theta2), velocity * np.sin(theta2)]
        # b2_start = Dot(np.dot(M, p2), radius=0.05).set_color(PINK)
        # b2 = Dot(np.dot(M, p2), radius=0.05).set_color(PINK)
        #
        # p3 = [1,1.5]
        # theta3 = PI / 3
        # v3 = [velocity * np.cos(theta3), velocity * np.sin(theta3)]
        # b3 = Dot(np.dot(M, p3), radius=0.05).set_color(BLUE)

        num_col = 10
        num_animation = 10
        L = 4*1.3
        Radius = 2*1.3
        #r1 = 4
        #r2 = 10

        render1 = billiard(p1, v1, num_col, L, Radius)
        #time1 = time(p1, v1, num_col, render1)/6
        #render2 = billiard(p2, v2, num_col, L, Radius)
        #time2 = time(p2, v2, num_col, render2)/6
        #print("tempi: ",time1, time2)
        # trace lascia la traccia, opacity can be an array
        trace1 = TracedPath(b1.get_center, stroke_opacity=0.3, stroke_color=b1.get_color(), stroke_width=2)
        #trace2 = TracedPath(b2.get_center, stroke_opacity=0.3, stroke_color=b2.get_color(), stroke_width=2)#, dissipating_time=0.5)
        #trace1 = VMobject()
        #trace2 = VMobject()
        #trace3 = VMobject()

        # filling the billiard
        rect = Rectangle(color=stadium_back, width=L, height=2*Radius).set_fill(stadium_back, opacity=1)
        stad1 = AnnularSector(color=stadium_back, fill_opacity=1, inner_radius=0, outer_radius=Radius, angle=-PI, start_angle=3 * PI / 2, arc_center=[-L / 2, 0, 0])
        stad2 = AnnularSector(color=stadium_back, fill_opacity=1, inner_radius=0, outer_radius=Radius, angle=-PI, start_angle=PI / 2, arc_center=[L / 2, 0, 0])
        self.add(rect, stad1, stad2)
        # drawing the billiard

        #self.add(b1, b2, trace1, trace2, b1_start, b2_start)
        self.add(trace1, b1_start)
        self.add(Line([-L / 2, Radius, 0], [L / 2, Radius, 0], stroke_width=5, color=stadium_bord))
        self.add(Line([-L / 2, -Radius, 0], [L / 2, -Radius, 0], stroke_width=5, color=stadium_bord))
        self.add(Arc(Radius, 3 * PI / 2, -PI, arc_center=[-L / 2, 0, 0], stroke_width=5, color=stadium_bord))
        self.add(Arc(Radius, PI / 2, -PI, arc_center=[L / 2, 0, 0], stroke_width=5, color=stadium_bord))

        num_start_balls=6
        centers = [stop(np.dot(reduc, render1[0]), np.dot(reduc, render1[1]), p) for p in linspace(0,0.2,num_start_balls)]
        opty = linspace(0.5,0,num_start_balls)
        starting_balls = VGroup(*[
           Circle(arc_center=centers[i], radius=r_ball1, color=col_orb1, fill_opacity=opty[i], stroke_width=0)
        for i in range(num_start_balls)])
        self.add(starting_balls)

        perc = 0.6

        Anim1 = Succession(
            *[
                AnimationGroup(
                    Create(Line(start=render1[i - 1], end=render1[i], stroke_width=2, color=b1.get_color())),
                    #MoveAlongPath(b1, Line(start=render1[i - 1], end=render1[i]), rate_func=linear)
                )
                for i in range(1, num_animation)
            ],
            AnimationGroup(
                Create(Arrow(start=render1[num_animation - 1],
                            end=stop(np.dot(reduc,render1[num_animation - 1]), np.dot(reduc, render1[num_animation]),perc),
                            stroke_width=2, color=b1.get_color(), buff=0, max_tip_length_to_length_ratio=r_ball1)
                            ),
                #MoveAlongPath(b1,
                #              Line(start=render1[num_animation - 1],
                #                   end=stop(np.dot(reduc,render1[num_animation-1]), np.dot(reduc, render1[num_animation]),perc))
                #              )
            )
        )
        #.set_color(PINK)
        # Anim2 = Succession(
        #     *[
        #         AnimationGroup(
        #             Create(Line(start=render2[i - 1], end=render2[i], stroke_width=2).set_color(PINK)),
        #             MoveAlongPath(b2, Line(start=render2[i - 1], end=render2[i]), rate_func=linear).set_run_time(
        #                 time(render2[i - 1], render2[i], 3 * velocity))
        #         )
        #         for i in range(1, num_animation)
        #     ],
        #     AnimationGroup(
        #         Create(Line(start=render2[num_animation - 1],
        #                     end=stop(np.dot(reduc,render2[num_animation - 1]), np.dot(reduc, render2[num_animation]),perc),
        #                     stroke_width=2, color=b2.get_color())),
        #         MoveAlongPath(b1,
        #                       Line(start=render1[num_animation - 1],
        #                            end=stop(np.dot(reduc,render2[num_animation-1]), np.dot(reduc, render2[num_animation]),perc))
        #                       )
        #     )
        # )

        for i in range(1, num_animation+1):
            print(render1[i-1], render1[i])

        #self.play(Anim1, Anim2)
        self.play(Anim1)



class Draw_image_multiple(Scene):
    def construct(self):
        self.camera.background_color = WHITE  # "#ece6e2"

        stadium_back = WHITE  # "#c2c4c6"#f2f4f6"#"#8598a7"#"#069fd9"#"#05014a"
        stadium_bord = "#353c42"
        #col_orb1 = "#0000ff"  # "#05014a"

        M = [[1, 0], [0, 1], [0, 0]]
        reduc = [[1, 0, 0], [0, 1, 0]]
        velocity = 1
        r_ball1 = 0.08

        Num_points = 6
        Max_val_binom = 20
        x_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        #y_rnd = np.random.binomial(Max_val_binom, 0.5, Num_points)
        #print(((x_rnd)/Max_val_binom-0.5)*0.3/0.5)
        # binomial distribution with ray of 0.3 and centered in 0
        x_rnd = ((x_rnd) / Max_val_binom - 0.5) * 0 / 0.5
        #y_rnd = ((y_rnd) / Max_val_binom - 0.5) * 0.5 / 0.5
        # use this line to set vertical points with same direction
        y_rnd = [a for a in linspace(-0.15,0.15,Num_points)]
        p_list = [np.add([-1, 1], [x_rnd[i], y_rnd[i]]) for i in range(Num_points)]
        #p_list = [-1, 0]
        # theta list is a list of angles between PI*a and PI*b where linspace(a,b,Num_points)
        theta_list = [PI*m for m in linspace(0, 0, Num_points)]
        v_list = [[velocity*np.cos(a), velocity*np.sin(a)] for a in theta_list]
        List = [] # list of lists of points

        num_col = 12
        num_animation = 12
        L = 7
        Radius = 3.2


        for l in range(Num_points):
            List.append(billiard(p_list[l], v_list[l], num_col, L, Radius))

        #b1 = Dot(np.dot(M,p1),radius=0.02).set_color(ORANGE)
        #k2 = VMobject()  #element to draw previous segment behind a ball
        dots = VGroup(*[Dot(np.dot(M, p_list[i]), radius=r_ball1) for i in range(Num_points)])
        dots.set_color_by_gradient(ORANGE, PINK)  #color by gradient
        #dots.set_color(BLUE)
        # Traccie dietro le palline, da usare SOLO CON POCHE PALLINE (MAX 15 con MAX 10 PALLINE)
        traces = VGroup(*[TracedPath(b.get_center, stroke_opacity=0.2, stroke_color=b.get_color(), stroke_width=2, dissipating_time=3) for b in dots])

        #traces = VGroup(*[VMobject() for i in range(Num_points)])

        self.add(traces, dots)#, traces)

        num_start_balls = 6
        for l in range(Num_points):
            centers = [stop(np.dot(reduc, List[l][0]), np.dot(reduc, List[l][1]), p) for p in
                       linspace(0, 0.2, num_start_balls)]
            opty = linspace(0.5, 0, num_start_balls)
            starting_balls = VGroup(*[
                Circle(arc_center=centers[i], radius=r_ball1, color=dots[l].get_color(), fill_opacity=opty[i], stroke_width=0)
                for i in range(num_start_balls)])
            self.add(starting_balls)

        perc = 0.6

        TotalAnim = AnimationGroup(
            *[
                Succession(
                    *[
                        AnimationGroup(
                            Create(Line(start=List[l][i - 1], end=List[l][i], stroke_width=2, color=dots[l].get_color())),
                            # MoveAlongPath(b1, Line(start=render1[i - 1], end=render1[i]), rate_func=linear)
                        )
                        for i in range(1, num_animation)
                    ],
                    AnimationGroup(
                        Create(Arrow(start=List[l][num_animation - 1],
                                     end=stop(np.dot(reduc, List[l][num_animation - 1]),
                                              np.dot(reduc, List[l][num_animation]), perc),
                                     stroke_width=2, color=dots[l].get_color(), buff=0,
                                     max_tip_length_to_length_ratio=0.04)
                               ),
                        # MoveAlongPath(b1,
                        #              Line(start=render1[num_animation - 1],
                        #                   end=stop(np.dot(reduc,render1[num_animation-1]), np.dot(reduc, render1[num_animation]),perc))
                        #              )
                    )
                )
                for l in range(Num_points)
            ]
        )
        self.play(TotalAnim)
        self.add(Line([-L / 2, Radius, 0], [L / 2, Radius, 0], stroke_width=6, color=stadium_bord))
        self.add(Line([-L / 2, -Radius, 0], [L / 2, -Radius, 0], stroke_width=6, color=stadium_bord))
        self.add(Arc(Radius, 3 * PI / 2, -PI, arc_center=[-L / 2, 0, 0], stroke_width=6, color=stadium_bord))
        self.add(Arc(Radius, PI / 2, -PI, arc_center=[L / 2, 0, 0], stroke_width=6, color=stadium_bord))

