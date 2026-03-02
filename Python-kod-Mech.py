


import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
#Input from Info that was in 
xA = 0.0
yA = 0.0

xB = 2.32      
yB = 1.11      
mu = 13.6       
T0 = 10.6500262060016411  #


def f(x, y):
    y1 = y[0]
    y2 = y[1]
    dy1dx = y2
    dy2dx = (mu / T0) * np.sqrt(1.0 + y2**2)
    return np.vstack((dy1dx, dy2dx))


def bc(ya, yb):
    return np.array([
        ya[0] - yA,   # y1(xA) - yA = 0
        yb[0] - yB    # y1(xB) - yB = 0
    ])  
x_mesh = np.linspace(xA, xB, 50)

# guess y as straight line between endpoints
y1_guess = yA + (yB - yA) * (x_mesh - xA) / (xB - xA)
# guess slope as constant
y2_guess = np.gradient(y1_guess, x_mesh)
y_guess = np.vstack((y1_guess, y2_guess))
sol = solve_bvp(f, bc, x_mesh, y_guess, tol=1e-8, max_nodes=10000)

print("Converged:", sol.success)
print("Status:", sol.status, sol.message)

x_plot = np.linspace(xA, xB, 400)
y_plot = sol.sol(x_plot)[0]
yprime_plot = sol.sol(x_plot)[1]



plt.figure()
plt.plot(x_plot, y_plot, label="Numerical BVP solution")
plt.plot([xA, xB], [yA, yB], "o", label="Mount points")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.legend()
plt.title("Catenary cable shape (numerical)")
plt.show()