from utils import calc_eigenvalues, calc_eigenvectors, plot_phase_space, system_dynamics
from scipy.integrate import odeint
import numpy as np

# Define your system's parameters
gamma = 0.2
eta = 0.1
delta = 0.4
tau = 0.3

# Define your system's matrix
A = np.array([[-eta, gamma], [tau, delta]])
# Calculate eigenvalues and eigenvectors
eigenvalues = calc_eigenvalues(A)
eigenvectors = calc_eigenvectors(A)

# Print eigenvalues and eigenvectors
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# Define initial conditions for C and P
C0 = 1.0
P0 = 1.0

# Define time points for which to solve for C and P
t = np.linspace(0,10)

# Solve system of differential equations to obtain values for C and P at each time point in t
sol = odeint(system_dynamics,[C0,P0],t,args=(gamma,eta,delta,tau))

# Define the grid for the phase space
c = np.linspace(-2, 2, 20)
p = np.linspace(-2, 2, 20)
C, P = np.meshgrid(c, p)

# Calculate the vector field
U = gamma*P - eta*C
V = delta*P + tau*C

# Plot phase space
plot_phase_space(C, P, U, V, title='Phase Space')