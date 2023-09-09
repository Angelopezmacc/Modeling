"""

* Universidad del Rosario
* Modeling of dynamical systems
! Lab 4
! Ángel López

"""

# * Se importan librerías
from utils import calc_eigenvalues, calc_eigenvectors, plot_phase_space, plot_phase_space_trajectory, system_dynamics, trajectory
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# * Definición de los parámetros del sistema
gamma = 1
eta = 3
delta = 1
tau = -1

# * Se crea la matriz del sistema y se calculan los eigenvalores y eigenvectores
A = np.array([[-eta, gamma], [tau, delta]])
eigenvalues = calc_eigenvalues(A)
eigenvectors = calc_eigenvectors(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# * Se definen condiciones iniciales
C0 = 1.0
P0 = 1.0
t = np.linspace(0,10)

# * Se resuleve el sistema
sol = odeint(system_dynamics,[C0,P0],t,args=(gamma,eta,delta,tau))

# * Definición dimensiones gráfica
c = np.linspace(-2, 2, 200)
p = np.linspace(-2, 2, 200)
C, P = np.meshgrid(c, p)
U = gamma*P - eta*C
V = delta*P + tau*C

# * Plot diagrama fase
plot_phase_space(C, P, U, V, title='Phase Space')

# * Plot diagrama de fase con trayectoria
n_steps = 100
cList, pList = trajectory(C0, P0, gamma, eta, delta, tau, n_steps)
plot_phase_space_trajectory(C, P, U, V, cList=cList, pList=pList, title='Phase Space with Trajectory')



# * Simulando la trayectoria
c_trajectory, p_trajectory = trajectory(C0, P0, gamma, eta, delta, tau, n_steps)

# * Plot de la trayectoria
plt.figure(figsize=(10, 6))
plt.plot(range(n_steps), c_trajectory, label='Coalition Strength (C)')
plt.plot(range(n_steps), p_trajectory, label="Parties' Internal Factors (P)")
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Trajectory of Coalition Strength and Parties Internal Factors')
plt.legend()
plt.grid(True)
plt.show()







