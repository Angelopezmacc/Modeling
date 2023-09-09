import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def calc_eigenvalues(matrix):
    """
    Calculates the eigenvalues of a square matrix.
    :param matrix: A square matrix represented as a 2D numpy array.
    :return: A 1D numpy array containing the eigenvalues of the matrix.
    """
    return np.linalg.eigvals(matrix)

def calc_eigenvectors(matrix):
    """
    Calculates the eigenvectors of a square matrix.
    :param matrix: A square matrix represented as a 2D numpy array.
    :return: A 2D numpy array where each column is an eigenvector of the matrix.
    """
    _, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors

def plot_phase_space(X, Y, U, V, title=None):
    """
    Plots the phase space of a dynamic system given its vector field.
    :param X: A 2D numpy array representing the x-coordinates of the vector field.
    :param Y: A 2D numpy array representing the y-coordinates of the vector field.
    :param U: A 2D numpy array representing the x-components of the vectors in the vector field.
    :param V: A 2D numpy array representing the y-components of the vectors in the vector field.
    :param title: An optional title for the plot.
    """
    plt.figure()
    
    # Plot the vector field using streamplot
    plt.streamplot(X, Y, U, V, color='r', linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
    
    if title:
        plt.title(title)
    plt.xlabel('C')
    plt.ylabel('P')
    plt.show()



def system_dynamics(y, t, gamma, eta, delta, tau):
     """
     Defines the system of differential equations that describe the dynamics of the coalition between SPD and CDU/CSU in Germany.

     dC/dt = γP − ηC (1)
     dP/dt = δP + τC (2)

     Where:
     • C: represents the strength or stability of the coalition between the parties.
     • P: represents the internal factors influencing the willingness of the parties to cooperate.
     • η, γ: are coefficients representing the relative impact of the coalition strength and the parties internal factors on 
       on coalition dynamics.
     • δ, τ : correspond to influence of coalition strength on parties’ internal dynamics and influence of public opinion on 
       parties’ willingness to cooperate.

     :param y: A list containing initial values for C and P
     :param t: An array containing time points for which to solve for y
     :param gamma: The relative impact of parties' internal factors on coalition dynamics
     :param eta: The relative impact of coalition strength on coalition dynamics
     :param delta: The influence of public opinion on parties' willingness to cooperate
     :param tau: The influence of coalition strength on parties' internal dynamics
     :return: Returns a list containing dC/dt and dP/dt
     """

     C,P = y
     dCdt = gamma*P - eta*C
     dPdt = delta*P + tau*C

     return [dCdt,dPdt]



     