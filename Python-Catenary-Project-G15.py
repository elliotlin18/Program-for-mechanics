import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.optimize import root
import matplotlib.pyplot as plt

# Cable and problem parameters given by table
xA = 0.0                     
yA = 0.0                      
xB = 2.32                     
yB = 1.11                     
l  = 3.45                     
mu = 13.6                    
T0 = 10.6500262060016411      


# define the ODE system for the catenary shape and the boundary conditions for the BVP solver
def ode_system(x, y):
    return [y[1], (mu / T0) * np.sqrt(1.0 + y[1]**2)]
# Boundary conditions: y(xA) = yA and y(xB) = yB
def boundary_conditions(ya, yb):
    return [ya[0] - yA, yb[0] - yB]

# Initial grid and linear guess
x_init = np.linspace(xA, xB, 20)
y_init = np.zeros((2, len(x_init)))
y_init[0] = np.linspace(yA, yB, len(x_init))
y_init[1] = (yB - yA) / (xB - xA)

# Numerical BVP solution
bvp_sol = solve_bvp(ode_system, boundary_conditions, x_init, y_init,
                    tol=1e-8, verbose=2)

# Evaluate numerical solution at dense grid for plotting and comparison
x = np.linspace(xA, xB, num=500)
y_num  = bvp_sol.sol(x)[0]     # y(x)  from BVP
dy_num = bvp_sol.sol(x)[1]     # y'(x) from BVP

# For task 3, plotting the numerical solution for the cable shape:
plt.figure()
plt.plot(x, y_num, label='Numerical (BVP)')
plt.plot([xA, xB], [yA, yB], 'ko', label='Mount points A, B')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title('Task 3 – Numerical cable shape (Group 15)')
plt.legend(loc='best')
plt.axis('equal')


# task 6 – Analytical solution for the cable shape

# The analytical solution for the catenary shape is given by:
def y_analytic(x, x0, y0):
    a = T0 / mu     # catenary constant
    return a * (np.cosh((x - x0) / a) - 1.0) + y0

# To find the parameters x0 and y0 of the lowest point, we use the boundary conditions at A and B:
def root_func(z):
    x0, y0 = z
    return [yA - y_analytic(xA, x0, y0),
            yB - y_analytic(xB, x0, y0)]
# A numerical root-finding method to solve for x0 and y0:
sol_root = root(root_func, [-0.5, -0.5], tol=1e-12)
x0, y0 = sol_root.x

print('--- Task 6 ---')
print(f'x0 = {x0:.6f} m  (x-coordinate of the lowest point)')
print(f'y0 = {y0:.6f} m  (y-coordinate of the lowest point)')
print(f'y_analytic(xA) = {y_analytic(xA, x0, y0):.8f}  (should be {yA})')
print(f'y_analytic(xB) = {y_analytic(xB, x0, y0):.8f}  (should be {yB})')

y_an  = y_analytic(x, x0, y0)
error = np.abs(y_num - y_an)
print(f'Max error between numerical and analytical: {np.max(error):.2e} m')

# --- Figure 2: Cable shape ---
plt.figure()
plt.plot(x, y_an,  label='Analytical')
plt.plot(x, y_num, linestyle='dashed', label='Numerical (BVP)')
plt.plot([xA, xB], [yA, yB], 'ko', label='Mount points A, B')
plt.plot([x0], [y0], 'bs', label='Lowest point $(x_0, y_0)$')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title('Task 6 – Cable shape: analytical vs. numerical (Group 15)')
plt.legend(loc='best')
plt.axis('equal')

# --- Figure 3: Absolute error ---
plt.figure()
plt.semilogy(x, error)
plt.xlabel('$x$ [m]')
plt.ylabel('$|y_\\mathrm{numerical} - y_\\mathrm{analytical}|$')
plt.title('Task 6 – Absolute error between numerical and analytical solution (Group 15)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# task 7 – Tension distribution along the cable
def T_analytic(x, x0):
    a = T0 / mu
    return T0 * np.cosh((x - x0) / a)

# The tension from the numerical solution is computed using T(x) = T0 * sqrt(1 + (dy/dx)^2):
T_an  = T_analytic(x, x0)
T_num = T0 * np.sqrt(1.0 + dy_num**2)

# Print results for Task 7
print('\n--- Task 7 ---')
print(f'T(x0) = {T_analytic(x0, x0):.4f} N  (= T0, minimum tension at lowest point)')
print(f'T(xA) = {T_analytic(xA, x0):.4f} N  (tension at mount point A)')
print(f'T(xB) = {T_analytic(xB, x0):.4f} N  (tension at mount point B)')

#Figure 4: Tension distribution
plt.figure()
plt.plot(x, T_an,  label='$T(x)$ analytical')
plt.plot(x, T_num, linestyle='dashed', label='$T(x)$ numerical')
plt.axhline(T0, color='gray', linestyle='dotted',
            label=f'$T_0 = {T0:.4f}$ N')
plt.xlabel('$x$ [m]')
plt.ylabel('$T$ [N]')
plt.title('Task 7 – Tension in the catenary cable (Group 15)')
plt.legend(loc='best')

# task 8 – Reaction forces at the mount points
# The reaction forces at the mount points can be computed from the tension and the slope of the
dydx_A = float(bvp_sol.sol(xA)[1])
dydx_B = float(bvp_sol.sol(xB)[1])

# Reaction forces at A (cable pulls left and downward, wall reacts right and upward)
F_Ax =  T0
F_Ay = -T0 * dydx_A
F_A  = np.sqrt(F_Ax**2 + F_Ay**2)

# Reaction forces at B (cable pulls right and downward, wall reacts left and upward)
F_Bx = -T0
F_By = -T0 * dydx_B
F_B  = np.sqrt(F_Bx**2 + F_By**2)

# Print results for Task 8
print('\n--- Task 8 ---')
print(f'dy/dx at A = {dydx_A:.6f}')
print(f'dy/dx at B = {dydx_B:.6f}')
print(f'Reaction force at A:  Fx = {F_Ax:.4f} N,  Fy = {F_Ay:.4f} N,  |F| = {F_A:.4f} N')
print(f'Reaction force at B:  Fx = {F_Bx:.4f} N,  Fy = {F_By:.4f} N,  |F| = {F_B:.4f} N')

#Task 9 – Total potential energy of the cable
# The potential energy of a cable segment is given by:
def integrand(x):
    y  = bvp_sol.sol(x)[0]
    dy = bvp_sol.sol(x)[1]
    return mu * y * np.sqrt(1.0 + dy**2)
# The total potential energy is the integral of the integrand along the cable:
Ep, err = quad(integrand, xA, xB)

# Print results for Task 9
print('\n--- Task 9 ---')
print(f'Total potential energy: Ep = {Ep:.6f} J')
print(f'Integration error estimate: {err:.2e} J')

plt.show()
 
