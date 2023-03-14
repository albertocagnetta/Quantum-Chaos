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
from scipy import interpolate
from scipy.interpolate import interp1d


from hyp_functions import *

# CONSTANTS
INF = float('inf')
EPS = 1e-12
EPSX = 1e-12


def arcs(point_list, geom_list, num_col):
    points = []
    for i in range(num_col):
        points.append(points_on_geodesic(point_list[2*i], point_list[2*i+1], [geom_list[i][0],0]))



def hyp_billiard(p, v, center, radius, line, num_col):
    """
    Simulate the movement of a point in a hyperbolic billiard
    :param p: starting point, 2D-Vector
    :param v: initial speed vector, 2D vector
    :param center: center of the circle of the modular surface (x,0), center=x a real number
    :param radius: radius of the circle of the modular surface
    :param line: x-coordinate of the vertical axis on the right of the modular surface
    :param num_col: number of collisions
    :return: [point_list, geom_list] where point_list is the list of the extrema of the geodesics of the orbit, geom_list is the list of the radii and centers
    """

    #v = [np.cos(alpha),np.sin(alpha)]
    J = np.array([[0, -1], [1, 0]])
    T = np.array([[1 , 1], [0, 1]])
    Tinv = np.array([[1,-1],[0,1]])
    A = matrix_direction(p, v)
    dir = 1
    new_p1 = p
    print("partenza e direzione:", new_p1, angle(v)*180/PI)
    new_v = v
    M = A #the matrix that gives the sequence
    alphabet = list('A')
    point_list = []
    geom_list = []
    point_list.append([p[0], p[1], 0])
    [c1, r1] = geod_from_point(p,v)
    print("matrice:", M)
    print("centro e raggio:", c1, r1)
    for i in range(num_col):
        print("volta", i+1)
        #[a1,a2,c1,r1]=geod_from_matrix(M)
        geom_list.append([c1, r1])
        #x_inter = [inters_geodesic_and_line_wall(new_p1, new_v, SX_WALL)[0][0],
        #           inters_geodesic_and_line_wall(new_p1, new_v, DX_WALL)[0][0],
        #           inters_geodesic_and_circled_wall(new_p1, new_v, C1, R1)[0][0]]
        x_inter = [inters_geodesic_and_line_wall(new_p1, new_v, -line)[0][0],
                   inters_geodesic_and_line_wall(new_p1, new_v, line)[0][0],
                   inters_geodesic_and_circled_wall(new_p1, new_v, center, radius)[0][0]]

        print("punti intersezione:", x_inter)
        if dir == 1:
            hit_wall = x_inter.index(min(filter(lambda a: a > new_p1[0]+EPSX, x_inter)))
            #hit_wall = times.index(min(x_inter))
        elif dir == -1:
            hit_wall = x_inter.index(max(filter(lambda a: a < new_p1[0]-EPSX, x_inter)))
            #hit_wall = times.index(min(x_inter))
        print('muro:', hit_wall)
        # print("nuova posizione", new_p[0], new_p[1])
        if hit_wall == 0:
            [new_p2, new_v] = inters_geodesic_and_line_wall(new_p1, new_v, -line)
            print("punto impatto:", new_p2)
            #new_v = inters_geodesic_and_line_wall(new_p1, new_v, SX_WALL)[1]
            point_list.append([new_p2[0], new_p2[1], 0])
            new_p1 = np.subtract(new_p2, [-1, 0])
            new_v = new_v
            print("nuova partenza e direzione:", new_p1, angle(new_v)/PI*180)
            point_list.append([new_p1[0], new_p1[1], 0])
            M = np.matmul(T,M)
            print("matrice:", M)
            [c1, r1] = geod_from_point(new_p1, new_v)
            print("centro e raggio:", c1, r1)
            dir = dir
            print("direzione:", dir)
            alphabet = ['T']+alphabet
        elif hit_wall == 1:
            [new_p2, new_v] = inters_geodesic_and_line_wall(new_p1, new_v, line)
            print("punto impatto:", new_p2)
            #new_v = inters_geodesic_and_line_wall(new_p1, new_v, DX_WALL)[1]
            point_list.append([new_p2[0], new_p2[1], 0])
            new_p1 = np.subtract(new_p2,[1,0])
            new_v = new_v
            print("nuova partenza e direzione:", new_p1, angle(new_v)/PI*180)
            point_list.append([new_p1[0], new_p1[1], 0])
            M = np.matmul(Tinv,M)
            print("matrice:", M)
            [c1, r1] = geod_from_point(new_p1, new_v)
            print("centro e raggio:", c1, r1)
            dir = dir
            print("direzione:", dir)
            alphabet = ['T^(-1)'] + alphabet
        else: #hit_wall == 2:
            [new_p2, new_v] = inters_geodesic_and_circled_wall(new_p1, new_v, center, radius)
            print("punto impatto:", new_p2)
            #new_v = inters_geodesic_and_circled_wall(new_p1, new_v, C1, R1)[1]
            point_list.append([new_p2[0], new_p2[1], 0])
            new_p1 = compl_to_2D(hyp_action(J,D2_to_compl(new_p2)))
            new_v = hyp_action_vel(new_p2, new_v, J)
            print("nuova partenza e direzione:", new_p1, angle(new_v) / PI * 180)
            point_list.append([new_p1[0], new_p1[1], 0])
            M = np.matmul(J,M)
            print("matrice:", M)
            [c1, r1] = geod_from_point(new_p1, new_v)
            print("centro e raggio:", c1, r1)
            if c1 != 0:
                dir = np.sign(c1)
            print("direzione:", dir)
            if new_p2[0] >= center:
                alphabet = ['J'] + alphabet
            else:
                alphabet = ['J^(-1)'] + alphabet
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(alphabet)
    point_list.pop()
    print(len(point_list))
    print(len(geom_list))
    #geom_list.pop()
    return [point_list, geom_list]






class Billiard(Scene):
    def construct(self):
        """
        It is possible to start from a matrix A which gives the initial position and speed, using the identification of
        the tangent space T^1 H with the group PSL2R
        :return: the orbit on the modular surface
        """

        reduction = np.array([[1, 0, 0], [0, 1, 0]])
        ampl = np.array([[1,0],[0,1],[0,0]])
        velocity = 1
        num_col = 12
        "Geometric data of the modular surface:"
        center = 0
        radius = 1
        xaxis = 0.5
        ########

        h = -3  # vertical shift

        "Starting point"
        #p1 = np.array([0, 3])
        #theta = PI / 2.5
        ## some periodic orbits (uncomment if you want this, but comment lines above)
        #k = 2
        #A = np.multiply(pow(k, -1 / 4) * 0.5, [[2 * (k + math.sqrt(k) - 1), 2 * (k - math.sqrt(k) - 1)],
        #                                       [k + math.sqrt(k), k - math.sqrt(k)]])
        # [p1, v1] = starting_from_M(A)
        ###########
        # p1 = np.array([0, 1.2])
        # theta = PI / 2.5
        ###########
        #p1 = np.array([-1/2, np.sqrt(0.5**2+1**2)])
        #theta = 0
        ###########
        #p1 = np.array([0, np.sqrt((1+0.5)**2+3/4)])
        #theta = 0
        v1 = np.array([velocity * np.cos(theta), velocity * np.sin(theta)])
        b1 = Dot((np.dot(ampl, p1)+[0,h,0]), radius=0.05).set_color(TEAL_D)#(BLUE_C)
        self.add(b1)
        ##########################
        "billiard simulation"
        [point_list, geom_list] = hyp_billiard(p1, v1, center, radius, xaxis, num_col)
        ##########################

        "Plot the domain"
        left = Line([-xaxis, h, 0], [-xaxis, 100, 0], color=GREEN)
        right = Line([xaxis, h, 0], [xaxis, 100, 0], color=GREEN)
        arc = Arc(1, PI / 3, PI / 3, arc_center=[center, h, 0], color=BLUE_E)
        #graph = axes.plot(lambda x: spl(x), x_range=[-1, 1], use_smoothing=True)
        self.add(left)
        self.add(right)
        self.add(arc)

        list_anim=[]
        for i in range(num_col):
            q1 = point_list[2*i]
            q2 = point_list[2*i+1]
            c = geom_list[i][0]
            r = geom_list[i][1]
            theta1 = angle(np.subtract(np.dot(reduction, q1), [c, 0]))
            theta2 = angle(np.subtract(np.dot(reduction, q2), [c, 0]))
            arc = Arc(r, theta1, theta2-theta1, arc_center=[c, h, 0], stroke_width=2, color=BLUE)
            #.set_run_time(r*(theta2-theta1))
            group = AnimationGroup(
                Create(arc, rate_func=linear).set_run_time(1.5*r*np.abs(theta2-theta1)),
                MoveAlongPath(b1, arc, rate_func=linear).set_run_time(1.5*r*np.abs(theta2-theta1))
            )
            list_anim.append(group)
            #self.play(MoveAlongPath(b1, Line([-0.5, h, 0], [-0.5, h, 0])))#Create(arc))
        #print(list_anim)
        Anim = Succession(*list_anim)
        self.play(Anim)#, rate_func=linear)#rate_functions.ease_out_sine)


























class Billiard3D(ThreeDScene):
    def construct(self):
        reduc = np.array([[1, 0, 0], [0, 1, 0]]) #from 3D vector to 2D
        ampl = np.array([[1, 0], [0, 1], [0, 0]]) #from 2D vector to 3D
        velocity = 1
        num_col = 24
        "Geometric data of the modular surface:"
        center = 0
        radius = 1
        xaxis = 0.5
        ########
        Radius = 2 #it's the radius of the sphere and, then, the scaling factor of the whole image
        scale_factor = 0.05  # dilatation from infinity to zero on the sphere: the circle of radius 1 on the plane corrisponds
        # not anymore to the middle disk of the sphere, but to a lower disk
        ######################################
        "initial position of the camera: vertical"
        self.set_camera_orientation(phi=0, theta=-PI / 2, distance=20)
        ax = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            z_range=[-2, 2, 0.5],
            axis_config={"include_tip": True},
            x_length=4,
            y_length=4,
            z_length=3
            # light_source=(-2, -2, 2)
        )
        sph_riem = Sphere(
            center=(0, 0, 0),
            radius=Radius,
            resolution=(30, 30),
            u_range=[0.001, PI - 0.001],
            v_range=[0, TAU]
        )
        sph_riem.set_opacity(0.2)
        sph_riem.set_color(BLUE_C)
        self.add(ax, sph_riem)
        self.wait(2)

        h = 0  # vertical shift

        "Starting point"
        p1 = np.array([0, 3])
        theta = PI / 2.5
        ## some periodic orbits (uncomment if you want this, but comment lines above)
        # k = 2
        # A = np.multiply(pow(k, -1 / 4) * 0.5, [[2 * (k + math.sqrt(k) - 1), 2 * (k - math.sqrt(k) - 1)],
        #                                       [k + math.sqrt(k), k - math.sqrt(k)]])
        # [p1, v1] = starting_from_M(A)
        ###########
        # p1 = np.array([0, 1.2])
        # theta = PI / 2.5
        ###########
        #p1 = np.array([-1/2, np.sqrt(0.5**2+1**2)])
        #theta = 0
        ###########
        # p1 = np.array([0, np.sqrt((1+0.5)**2+3/4)])
        # theta = 0
        v1 = np.array([velocity * np.cos(theta), velocity * np.sin(theta)])
        b1 = Dot((np.multiply(Radius, np.dot(ampl, p1)) + [0, h, 0]), radius=0.05).set_color(
            YELLOW_C)  #(BLUE_C)  # (BLUE_C)
        self.add(b1)




        left = Line([-xaxis*Radius, h, 0], [-xaxis*Radius, 10, 0])
        right = Line([xaxis*Radius, h, 0], [xaxis*Radius, 10, 0])
        arc = Arc(Radius*radius, PI / 3, PI / 3, arc_center=[Radius*center, h, 0])
        self.add(left)
        self.add(right)
        self.add(arc)
        #self.play(Wait(run_time=3))
        self.play(b1.animate())
        self.wait(1)
        self.move_camera(phi=40 * DEGREES, theta=45 * DEGREES)
        self.wait(2)


        ##############################
        ##########################
        "billiard simulation"
        [point_list, geom_list] = hyp_billiard(p1, v1, center, radius, xaxis, num_col)
        ##########################
        # computing points along the orbit, to be interpolated
        sequence_on_modular = []
        for i in range(num_col):
            sequence_on_modular.append(
                points_on_geodesic(np.dot(reduc, point_list[2 * i]), np.dot(reduc, point_list[2 * i + 1]),
                                   [geom_list[i][0], 0])
            )
        x_list = []
        y_list = []
        z_list = []
        orbit2d_list = []
        orbit3d_list = []

        for l in sequence_on_modular:
            for q in l:
                orbit2d_list.append(q)
                q3d = from_C_to_sphere(scale_factor*kleinj(D2_to_compl(q)), Radius)
                #q = compl_to_2D(kleinj(D2_to_compl(q)))
                q3d = [np.float64(q3d[0]), np.float64(q3d[1]), np.float64(q3d[2])]
                #q = [np.float64(q[0]), np.float64(q[1])]
                x_list.append(q3d[0])
                y_list.append(q3d[1])
                z_list.append(q3d[2])
                orbit3d_list.append(q3d)
                #orbit3d_list.append(self.wait(0.3))
                #print("punto:",q3d)
                #self.add(Dot(np.dot(ampl,q), radius=0.02).set_color(PINK))
        self.wait(3)
        #############
        " 2D and 3D point orbits"
        orbit2d = VGroup(*[Dot(np.add(np.multiply(Radius,np.dot(ampl,q)),[0,h,0]), radius=0.02) for q in orbit2d_list])
        orbit = VGroup(*[Dot(q, radius=0.02) for q in orbit3d_list])
        orbit2d.set_color_by_gradient(PINK, BLUE_E, YELLOW)
        orbit.set_color_by_gradient(PINK, BLUE_E, YELLOW)
        #############################
        " Static orbit points on the plane and on the sphere "
        # self.add(orbit2d)
        # self.add(orbit)
        # self.wait(4)
        " Animated sequence of points on the sphere: uncomment to do that, but then comment above lines (two add commands)"
        # stop = Wait(run_time=0.2)
        # moving_point = Succession(
        #     *[
        #         AnimationGroup(
        #             Create(q)#,run_time=0.08),
        #             #stop
        #         )
        #         for q in orbit
        #     ]
        # , run_time=30, rate_func=rate_functions.linear)
        # self.play(moving_point)#, rate_func=rate_functions.smooth)
        #self.wait(4)

        #####################################
        " Animated sequence of points on the sphere and the plane:"
        stop = Wait(run_time=0.2)
        moving_point3D = Succession(
            *[
                AnimationGroup(
                    Create(q)#,run_time=0.08),
                    #stop
                )
                for q in orbit
            ]
        , run_time=90, rate_func=rate_functions.linear)
        moving_point2D = Succession(
            *[
                AnimationGroup(
                    Create(q)  # ,run_time=0.08),
                    # stop
                )
                for q in orbit2d
            ]
            , run_time=90, rate_func=rate_functions.linear)
        self.play(moving_point3D, moving_point2D)#, rate_func=rate_functions.smooth)
        self.wait(4)
        #####################################

        ##  ATTENTION: comment the above dot and active the following lines to plot moving point along orbit
        ## NOT WORKING
        # tck, u = interpolate.splprep([x_list, y_list, z_list], s=2)
        # sample = 2*num_col*100 #the 100 is the number points in geodesic points function
        # eval_pts = np.linspace(0.0, 1, sample)
        # t = ValueTracker()
        # smooth_tx, smooth_ty, smooth_tz = interpolate.splev(eval_pts, tck)
        #
        # initial_point = [ax.coords_to_point(t.get_value(), interpolate.splev(t.get_value(), tck)[1])]
        # point = Dot(point=initial_point, color=PINK, radius=0.04)
        #
        # path = VMobject()
        # path.set_points_as_corners([point.get_center(), point.get_center()])
        #
        # def update_path(path):
        #     previous_path = path.copy()
        #     previous_path.add_points_as_corners([dot.get_center()])
        #     path.become(previous_path)
        #
        # path.add_updater(update_path)
        # path.set_color(PINK)#dot.get_color())
        # # path.set_color_by_gradient(PINK, GREEN, YELLOW)
        # path.set_style(stroke_width=1.5)
        # dot.add_updater(lambda x: x.move_to(ax.coords_to_point(smooth_tx[int(t.get_value())],
        #                                                        smooth_ty[int(t.get_value())],
        #                                                        smooth_tz[int(t.get_value())]
        #                                                        )))
        #
        # self.add(path)
        # self.wait()
        # self.play(t.animate.set_value(sample - 1), run_time=30, rate_func=linear)
        # self.wait(6)








