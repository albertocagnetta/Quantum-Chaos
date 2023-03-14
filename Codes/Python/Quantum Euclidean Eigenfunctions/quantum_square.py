import numpy as np
import math
import matplotlib.pyplot as plt
import mpmath as mp
from mpmath import besselj, findroot
from scipy import sparse
from scipy.sparse import linalg as sla
from numpy import sin, cos, pi, linspace

PI = math.pi
pi = 3

def norm(v):
    return np.sqrt(np.dot(v, v))

def angle(v): #gives the angle of a vector
    if v[0] == 0:
        if v[1] > 0:
            theta = PI/2
        else:
            theta = -PI/2
    elif v[1] == 0:
        if v[0] > 0:
            theta = 0
        else:
            theta = PI
    elif v[0] > 0:
        theta = math.atan(v[1]/v[0])
    else:
        theta = math.atan(v[1]/v[0])-PI*np.sign(v[1]/v[0])
    return theta

def eig_square(x, y, n, m, L, c, e):
    return (np.sin(x*n*PI/L)*np.sin(y*m*PI/L))**2

def eig_disk(x, y, k, zero, e):
    r = norm(np.array([x,y]))
    theta = angle(np.array([x,y]))
    return ((np.sin(k*theta)+np.cos(k*theta))*besselj(k, r*zero))**e #(np.sin(x*n*PI/L)*np.sin(y*m*PI/L))


def square_eigenfunctions(c,L,num_bes,neigs,E):
    Nx = L*200
    Ny = Nx
    xmin = c[0]-L/2
    xmax = c[0]+L/2
    ymin = c[1]-L/2
    ymax = c[1]+L/2
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    plt.figure(figsize=(20, 20))
    for n in range(num_bes):
        for m in range(neigs):
            plt.subplot(neigs, neigs, neigs*n + m+1)
            x1, y1 = np.meshgrid(x,y)  # accorgmenti grafici
            #print(x1,y1)
            y1 = np.flipud(y1)
            Z = eig_square(x1, y1, n+1, m+1, L, c, 2)
            plt.pcolormesh(x1, y1, Z, cmap='plasma')  # o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
    plt.savefig("quantum_square.png", bbox_inches='tight')
    plt.show()


def disk_eigenfunctions(R,num_bes,neigs,E):
    Nx = R*300
    Ny = Nx
    xmin = -R
    xmax = R
    ymin = -R
    ymax = R
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]
    l_roots = []
    l_eigs = []
    roots = []
    eigs = []
    bessels = []
    for k in range(num_bes):
        J = lambda t: besselj(k,t)
        bessels.append(J)
        for h in range(neigs):
            roots.append(np.float64(findroot(J, np.sqrt(E) + k + pi*h)))
            #print("start:", np.sqrt(E) + 2*pi*h)
            eigs.append(np.float64(findroot(J, np.sqrt(E) + k + pi*h)**2))
        l_roots.append(roots)
        l_eigs.append(eigs)
        roots = []
        eigs = []
        print("Radici trovate per aut funz k", k, l_roots[k])
        print("Autovalori per aut funz k",k,l_eigs[k])
    plt.figure(figsize=(30, 20))
    for k in range(num_bes):
        for n in range(neigs):
            plt.subplot(num_bes, neigs, num_bes*k + n+1)
            x1, y1 = np.meshgrid(x,y)  # accorgmenti grafici
            #print(x1,y1)
            y1 = np.flipud(y1)
            Z = np.zeros((len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    if x[i]**2+y[j]**2 < R**2:
                        Z[i, j] = eig_disk(x[i],y[j], k, l_roots[k][n], 2)
            #Z = eig_disk(x1,y1,k,l_roots[k][n],2)
            plt.pcolormesh(x1, y1, Z, cmap='jet')  # o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
            magnitude = 1.01
            arc_angles = np.linspace(0, 2 * PI, 300)
            circle = [magnitude * R * np.cos(arc_angles), magnitude * R * np.sin(arc_angles)]
            plt.plot(circle[0], circle[1], color='orange', lw=1)
    plt.show()


L = 1
c = np.array([L/2, L/2])

E1 = 1

num_bes = 4

square_eigenfunctions(c,L,num_bes,6,E1)

R = 1
num_bes = 3
E2 = 2

#disk_eigenfunctions(R,num_bes,6,E2)




