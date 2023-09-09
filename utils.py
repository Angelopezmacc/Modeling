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




     