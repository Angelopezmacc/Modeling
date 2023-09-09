"""

* Universidad del Rosario
* Modeling of dynamical systems
! Lab 4
! Ángel López

"""

# Se importan librerías
# ******************************************
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy as sp
# ******************************************

# ! Función que calcula eigenvalores
def calc_eigenvalues(matrix):

    return np.linalg.eigvals(matrix)

# * --------------------------------------------- *
# ! Función que calcula eigenvectores
def calc_eigenvectors(matrix):

    _, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors

# * --------------------------------------------- *
# ! Función que crea el diagrama de fase
def plot_phase_space(X, Y, U, V, title=None):
    plt.figure() 
    plt.streamplot(X, Y, U, V, color='r', linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
    
    if title:
        plt.title(title)
    plt.xlabel('C')
    plt.ylabel('P')
    plt.show()

# * --------------------------------------------- *
# ! Función que crea el diagrama de fase con la trayectoria incluida
def plot_phase_space_trajectory(X, Y, U, V, cList=None, pList=None, title=None):
    plt.figure()
    plt.streamplot(X, Y, U, V, color='r', linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
    if cList is not None and pList is not None:
        plt.plot(cList, pList, color='b')
    
    if title:
        plt.title(title)
    plt.xlabel('C')
    plt.ylabel('P')
    plt.show()

# * --------------------------------------------- *
# ! Función que define el sistema como fue descrito en el enunciado
def system_dynamics(y, t, gamma, eta, delta, tau):

     C,P = y
     dCdt = gamma*P - eta*C
     dPdt = delta*P + tau*C

     return [dCdt,dPdt]

# * --------------------------------------------- *
# ! Función que calcula la trayectoria (función dada en el enunciado que fue completada)
def trajectory(c_0, p_0, gamma, eta, delta, tau, n_steps: int):
    cList = [c_0]
    pList = [p_0]

    for t in range(1, n_steps):
        c_1 = gamma * pList[-1] - eta * cList[-1]
        p_1 = delta * pList[-1] + tau * cList[-1]

        cList.append(c_1)
        pList.append(p_1)

    return cList, pList

def general_solution(V1, V2, gamma, eta, delta, tau):
    # Definir las variables
    t = sp.symbols('t')
    alpha = sp.symbols('alpha')
    beta = sp.symbols('beta')

    # Descomponer los eigenvectores en sus componentes C y P
    V1_c = V1[0]
    V1_p = V1[1]
    
    V2_c = V2[0]
    V2_p = V2[1]

    # * Se crea la matriz del sistema y se calculan los eigenvalores y eigenvectores
    A = np.array([[-eta, gamma], [tau, delta]])
    eigenvalues = calc_eigenvalues(A)

    lambda1 = eigenvalues[0]  # Primer valor propio
    lambda2 = eigenvalues[1]  # Segundo valor propio

    # Crear la solución general
    c_solution = f"C(t) = {alpha} * e^({lambda1} * t) * {V1_c} + {beta} * e^({lambda2} * t) * {V2_c}"
    p_solution = f"P(t) = {alpha} * e^({lambda1} * t) * {V1_p} + {beta} * e^({lambda2} * t) * {V2_p}"

    return c_solution, p_solution





     