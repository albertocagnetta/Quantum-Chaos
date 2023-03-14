# code freely inspired

import pygame
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
from numpy import sin, cos, pi, linspace

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
from numpy import sin, cos, pi, linspace

PI = math.pi

def norm(v):
    return np.sqrt(np.dot(v, v))

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
        theta = math.atan(v[1]/v[0])
    else:
        theta = math.atan(v[1]/v[0])-PI*np.sign(v[1]/v[0])
    return theta


def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, params,
                  neigs, E0=0.0, findpsi=False):
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    V = Vfun2D(x, y, params)

    # Create the 2D Hamiltonian matrix
    # First, the derivatives in the x direction.
    # Note that instead of using arrays, we use sparse matrices
    # in order to reduce computational resource consumption.
    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / (dx ** 2)  # Next, the derivatives in the y direction.
    Hy = sparse.lil_matrix(2 * np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i, i + 1] = -1
        Hy[i + 1, i] = -1
    Hy = Hy / (dy ** 2)  # Combine both x and y Hilbert spaces using Kronecker products.
    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)

    # Re-convert to sparse matrix lil form.
    H = H.tolil()  # And add the potential energy.
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]

        # Convert to sparse matrix csc form,
    # and solve the eigenvalue problem
    H = H.tocsc()
    [evl, evt] = sla.eigs(H, k=neigs, sigma=E0)

    if findpsi == False:
        return evl
    else:
        return evl, evt, x, y

def eval_wavefunctions(xmin, xmax, Nx,ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi):

    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny,
                      Vfun, params, neigs, E0, findpsi)  # Get eigen energies

    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    evl = H[0]
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i, j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))  # Get eigen wave functions
    evt = H[1]
    plt.figure(figsize=(25, 20))
    magnitude = 1.01
    arc_angles = np.linspace(0 * pi / 3, 3 * pi / 3, 100)
    points_y = np.linspace(-magnitude*(ymax - (xmax - xmin) / 2), magnitude*(ymax - (xmax - xmin) / 2), 200)
    l = len(points_y)
    arc_xs = magnitude * (xmax - xmin) * cos(arc_angles) / 2 + magnitude * (xmax + xmin) / 2
    arc_ys = magnitude * (ymax - (xmax - xmin) / 2) + magnitude * (xmax - xmin) * sin(arc_angles) / 2
    side_ly = points_y
    side_lx = [magnitude * xmin for i in range(l)]
    side_ry = points_y
    side_rx = [magnitude * xmax for i in range(l)]
    under_arc_xs = -magnitude * (xmax - xmin) * cos(arc_angles) / 2 + magnitude * (xmax + xmin) / 2
    under_arc_ys = -magnitude * (ymax - (xmax - xmin) / 2) - magnitude * (xmax - xmin) * sin(arc_angles) / 2
    # Unpack the vector into 2 dimensions for plotting:
    for n in range(neigs):
        psi = evt[:, n]
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI) ** 2
        if neigs > 6:
            k = int(neigs/9)
            plt.subplot(k+1, 9, n+1)
            y, x = np.mgrid[
                slice(-Ny * dx / 2, Ny * dx / 2, dx), slice(-Nx * dx / 2, Nx * dx / 2, dx)]  # accorgmenti grafici
            plt.pcolormesh(x, y, np.flipud(PSI), cmap='plasma') #o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
            #plt.axis()
            plt.plot(arc_xs, arc_ys, color='orange', lw=2)
            plt.plot(under_arc_xs, under_arc_ys, color='orange', lw=2)
            plt.plot(side_rx, side_ry, color='orange', lw=2)
            plt.plot(side_lx, side_ly, color='orange', lw=2)
        else:
            plt.subplot(1, neigs, n + 1)
            y, x = np.mgrid[
                slice(-Ny * dx / 2, Ny * dx / 2, dx), slice(-Nx * dx / 2, Nx * dx / 2, dx)]  # accorgmenti grafici
            plt.pcolormesh(x, y, np.flipud(PSI), cmap='binary')  # binary, jet, inferno, plasma, viridis, gnuplot, gnuplot2, turbo mode
            plt.axis('equal')
            plt.axis('off')
            # plt.axis()
            plt.plot(arc_xs, arc_ys, color='orange', lw=2)
            plt.plot(under_arc_xs, under_arc_ys, color='orange', lw=2)
            plt.plot(side_rx, side_ry, color='orange', lw=2)
            plt.plot(side_lx, side_ly, color='orange', lw=2)
    # plt.gca().annotate('Arc', xy=(1.5, 0.4), xycoords='data', fontsize=10, rotation = 120)
    plt.savefig("quantum.png", bbox_inches='tight')
    plt.show()

def twoD_to_oneD(Nx, Ny, F):
    # From a 2D matrix F return a 1D vector V.
    V = np.zeros(Nx * Ny)
    vindex = 0
    for i in range(Ny):
        for j in range(Nx):
            V[vindex] = F[i, j]
            vindex = vindex + 1
    return V

def oneD_to_twoD(Nx, Ny, psi):
    # From a 1D vector psi return a 2D matrix PSI.
    vindex = 0
    PSI = np.zeros([Ny, Nx], dtype='complex')
    for i in range(Ny):
        for j in range(Nx):
            PSI[i, j] = psi[vindex]
            vindex = vindex + 1
    return PSI

def Vfun(X, Y, params):
    R = params[0]  # stadium radius
    L = params[1]  # stadium length
    V0 = params[2]  # stadium wall potential    # Stadium potential function.
    Nx = len(X)
    Ny = len(Y)
    [x, y] = np.meshgrid(X, Y)
    F = np.zeros([Ny, Nx])
    for i in range(Nx):
        for j in range(Ny):
            if abs(X[i]) == R or abs(Y[j]) == R + 0.5 * L:
                F[j, i] = V0
            cond_0 = (abs(Y[j]) - 0.5 * L) > 0
            cond_1 = np.sqrt((abs(Y[j]) - 0.5 * L) ** 2 + X[i] ** 2) >= R
            if cond_0 and cond_1:
                F[j, i] = V0  # Fold the 2D matrix to a 1D array.
    V = twoD_to_oneD(Nx, Ny, F)
    return V



def stadium_wavefunctions_plot(R , L, V0, neigs, E0 ,findpsi):
    # R = stadium radius
    # L = stadium length
    # V0 = stadium wall potential
    ymin = -0.5 * L - R
    ymax = 0.5 * L + R
    xmin = -R
    xmax = R
    params = [R, L, V0]
    Ny = 500
    Nx = int(Ny * 2 * R / (2.0 * R + L))
    eval_wavefunctions(xmin, xmax, Nx, ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi)





stadium_wavefunctions_plot(1, 2, 1e6, 36, 1000, True)



######################################################################################################


def V_card(X, Y, params):
    R = params[0]  # cardioid radius
    V0 = params[1]  # cardioid wall potential
    Nx = len(X)
    Ny = len(Y)
    [x, y] = np.meshgrid(X, Y)
    F = np.zeros([Ny, Nx])
    for i in range(Nx):
        for j in range(Ny):
            p = np.array([X[i],Y[j]])
            theta = angle(p)
            if norm(p) > R*(1-np.cos(theta)):
                F[j, i] = V0
    V = twoD_to_oneD(Nx, Ny, F)
    return V


def eval_card_wavefunctions(xmin, xmax, Nx,ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi):

    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny,
                      Vfun, params, neigs, E0, findpsi)  # Get eigen energies

    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    evl = H[0]
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i, j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))  # Get eigen wave functions
    evt = H[1]
    plt.figure(figsize=(30, 5))
    magnitude = 1.001
    angles = np.linspace(0, 2*PI, 2000)
    k = -xmin/(-xmin+xmax)
    x_card = (-xmin-xmax)/2 + magnitude*params[0]*(1-np.cos(angles))*np.cos(angles)
    y_card = magnitude*params[0]*(1-np.cos(angles))*np.sin(angles)
    # Unpack the vector into 2 dimensions for plotting:
    for n in range(neigs):
        psi = evt[:, n]
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI) ** 2
        if neigs > 6:
            plt.subplot(2, int(neigs / 2), n + 1)
            y, x = np.mgrid[
                slice(-Ny * dx / 2, Ny * dx / 2, dx), slice(-Nx * dx / 2, Nx * dx / 2, dx)]  # accorgmenti grafici
            plt.pcolormesh(x, y, np.flipud(PSI), cmap='jet') #o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
            #plt.axis()
            plt.plot(x_card, y_card, color='orange', lw=1)
        else:
            plt.subplot(1, neigs, n + 1)
            y, x = np.mgrid[
                slice(-Ny * dx / 2, Ny * dx / 2, dx), slice(-Nx * dx / 2, Nx * dx / 2, dx)]  # accorgmenti grafici
            plt.pcolormesh(x, y, np.flipud(PSI), cmap='binary')  # o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
            # plt.axis()
            plt.plot(x_card, y_card, color='orange', lw=1)
    # plt.gca().annotate('Arc', xy=(1.5, 0.4), xycoords='data', fontsize=10, rotation = 120)
    plt.savefig("quantum.png", bbox_inches='tight')
    plt.show()


def cardioid_wavefunctions_plot(R , V0, neigs, E0 ,findpsi):
    # R = stadium radius
    # L = stadium length
    # V0 = stadium wall potential
    ymin = 1.5*R
    ymax = -1.5 * R
    xmin = -2.5*R
    xmax = 0.5*R
    params = [R, V0]
    Ny = 400
    Nx = 400
    eval_card_wavefunctions(xmin, xmax, Nx, ymin, ymax, Ny, V_card, params, neigs, E0, findpsi)



#cardioid_wavefunctions_plot(5, 1e6, 24, 10, True)




########################################################################################################




def V_square(X, Y, params):
    L = params[0]  # cardioid radius
    V0 = params[1]  # cardioid wall potential
    Nx = len(X)
    Ny = len(Y)
    [x, y] = np.meshgrid(X, Y)
    F = np.zeros([Ny, Nx])
    for i in range(Nx):
        for j in range(Ny):
            p = np.array([X[i],Y[j]])
            if np.abs(X[i])>L/2 or np.abs(Y[j])>L/2:
                F[j, i] = V0
    V = twoD_to_oneD(Nx, Ny, F)
    return V


def eval_square_wavefunctions(xmin, xmax, Nx,ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi):

    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny,
                      Vfun, params, neigs, E0, findpsi)  # Get eigen energies

    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]
    L = xmax - xmin

    evl = H[0]
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i, j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))  # Get eigen wave functions
    evt = H[1]
    plt.figure(figsize=(30, 5))
    magnitude = 1.001
    line = np.linspace(-L/2, L/2, 200)
    xd = [magnitude*xmax for l in line]
    xs = [magnitude*xmin for l in line]
    yu = [magnitude*ymax for l in line]
    yd = [magnitude*ymin for l in line]
    # Unpack the vector into 2 dimensions for plotting:
    for n in range(neigs):
        psi = evt[:, n]
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI) ** 2
        if neigs > 6:
            plt.subplot(2, int(neigs / 2), n + 1)
            y, x = np.mgrid[
                slice(-Ny * dx / 2, Ny * dx / 2, dx), slice(-Nx * dx / 2, Nx * dx / 2, dx)]  # accorgmenti grafici
            plt.pcolormesh(x, y, np.flipud(PSI), cmap='jet') #o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
            #plt.axis()
            plt.plot(xd, line, color='orange', lw=1)
            plt.plot(xs, line, color='orange', lw=1)
            plt.plot(line, yu, color='orange', lw=1)
            plt.plot(line, yd, color='orange', lw=1)
        else:
            plt.subplot(1, neigs, n + 1)
            y, x = np.mgrid[
                slice(-Ny * dx / 2, Ny * dx / 2, dx), slice(-Nx * dx / 2, Nx * dx / 2, dx)]  # accorgmenti grafici
            plt.pcolormesh(x, y, np.flipud(PSI), cmap='binary')  # o binary o jet o inferno mode
            plt.axis('equal')
            plt.axis('off')
            # plt.axis()
            #plt.plot(x_card, y_card, color='orange', lw=1)
    # plt.gca().annotate('Arc', xy=(1.5, 0.4), xycoords='data', fontsize=10, rotation = 120)
    plt.savefig("square_quant.png", bbox_inches='tight')
    plt.show()


def square_wavefunctions_plot(L , V0, neigs, E0 ,findpsi):
    # R = stadium radius
    # L = stadium length
    # V0 = stadium wall potential
    dl = 0.1
    ymin = -L/2-dl
    ymax = L/2+dl
    xmin = -L/2-dl
    xmax = L/2+dl
    params = [L, V0]
    Ny = 400
    Nx = Ny
    eval_square_wavefunctions(xmin, xmax, Nx, ymin, ymax, Ny, V_square, params, neigs, E0, findpsi)


#square_wavefunctions_plot(2, 1e6, 24, 0.1, True)