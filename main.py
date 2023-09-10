"""

* Universidad del Rosario
* Modeling of dynamical systems
! Lab 4
! Ángel López

"""

# * Se importan librerías
from utils import calc_eigenvalues, calc_eigenvectors, plot_phase_space, \
plot_phase_space_trajectory, system_dynamics, trajectory, general_solution
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# * Definición de los parámetros del sistema

# 3.1
# gamma = 1
# eta = 3
# delta = 1
# tau = -1

# 3.2
# gamma = -2 #  Por la falta de cohesión para colaborar
# eta = 0.2 # No hay diferencias políticas significativas
# delta = 0 # La estabilidad de la coalición no afecta la voluntad de cooperar de las partes
# tau = -0.4 # Factor externo ligeramente desfavorable

# 3.3
# gamma = 3 #  Por la buena cohesión para colaborar
# eta = 0.2 # No hay diferencias políticas significativas
# delta = 0 # La estabilidad de la coalición no afecta la voluntad de cooperar de las partes
# tau = -0.3 # Factor externo ligeramente desfavorable

# 3.4
# gamma = 3 #  Por la falta de cohesión para colaborar
# eta = -5 # Hay diferencias políticas significativas
# delta = -0.7 # La estabilidad de la coalición tiene un fuerte impacto en la voluntad de cooperar de las partes
# tau = -2.7 # Influencias externas desfavorables desfavorable

# 3.5
gamma = 3 #  Por la buena cohesión para colaborar
eta = -0.3 # Hay diferencias políticas relativamente bajas
delta = 2.5 # La estabilidad de la coalición tiene un fuerte impacto positivo en la voluntad de cooperar de las partes
tau = -3 # Influencias externas desfavorables desfavorable

# * Se crea la matriz del sistema y se calculan los eigenvalores y eigenvectores
A = np.array([[-eta, gamma], [tau, delta]])
#print(A)
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

# * Mostrando la solución general
V1 = eigenvectors[:, 0]
V2 = eigenvectors[:, 1]
c_solution, p_solution = general_solution(V1, V2, gamma, eta, delta, tau)
print("General solution for C(t):", c_solution)
print("General solution for P(t):", p_solution)











