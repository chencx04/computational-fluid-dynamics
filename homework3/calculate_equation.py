import numpy as np
import matplotlib.pyplot as plt

class SolveEquation:
    def upwind(self, lam, nx, nt):
        h = 1 / (nx - 1)
        tau = h * lam
        u = np.zeros((nx, nt+1))
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        for i in range(nt):
            for j in range(nx):
                if j != 0:
                    u[j, i + 1] = (1-lam)*u[j, i] + lam*u[j-1, i]
                else:
                    u[j, i + 1] = (1-lam)*u[j, i] + lam*u[nx - 2, i]
        return u
    
    def lax(self, lam, nx, nt):
        h = 1 / (nx - 1)
        tau = h * lam
        u = np.zeros((nx, nt+1))
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        for i in range(nt):
            for j in range(nx):
                if j != 0 and j != nx-1:
                    u[j, i + 1] = (1-lam)/2*u[j+1, i]+(1+lam)/2*u[j-1,i]
                elif j == nx-1:
                    u[j, i + 1] = (1-lam)/2*u[1, i]+(1+lam)/2*u[j-1,i]
                else:
                    u[j, i + 1] = (1-lam)/2*u[j+1, i]+(1+lam)/2*u[nx-2,i]
        return u
    
    def lax_wendroff(self, lam, nx, nt):
        h = 1 / (nx - 1)
        tau = h * lam
        u = np.zeros((nx, nt+1))
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        for i in range(nt):
            for j in range(nx):
                if j != 0 and j != nx-1:
                    u[j, i + 1] = (1-lam**2)*u[j, i]-(lam/2)*(1-lam)*u[j+1,i]+(lam/2)*(1+lam)*u[j-1,i]
                elif j == nx-1:
                    u[j, i + 1] = (1-lam**2)*u[j, i]-(lam/2)*(1-lam)*u[1,i]+(lam/2)*(1+lam)*u[j-1,i]
                else:
                    u[j, i + 1] = (1-lam**2)*u[j, i]-(lam/2)*(1-lam)*u[j+1,i]+(lam/2)*(1+lam)*u[nx-2,i]
        return u

    def mac_cormack(self, lam, nx, nt):
        h = 1 / (nx - 1)
        tau = h * lam
        u = np.zeros((nx, nt+1))
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        u_bar = np.zeros((nx, nt+1))
        for i in range(nt):
            for j in range(nx):
                if j != nx-1:
                    u_bar[j, i+1] = (1+lam)*u[j,i]-lam*u[j+1,i]
                else:
                    u_bar[j, i+1] = (1+lam)*u[j,i]-lam*u[1,i]
            for j in range(nx):
                if j !=0 :
                    u[j, i+1] = (u[j,i]+u_bar[j,i+1])/2 - lam*(u_bar[j,i+1]-u_bar[j-1,i+1])/2
                else:
                    u[j, i+1] = (u[j,i]+u_bar[j,i+1])/2 - lam*(u_bar[j,i+1]-u_bar[nx-2,i+1])/2
        return u

    def exact_solution(self, lam, nx, nt):
        h = 1 / (nx - 1)
        tau = h * lam
        u = np.zeros((nx, nt+1))
        for i in range(nt+1):
            for j in range(nx):
                u[j, i] = np.sin(2*np.pi*(j*h-i*tau))
        return u
    
    def error(self, u_approx, u_exact):
        error = np.abs(u_approx - u_exact)
        return error

        

solve_equation = SolveEquation()
lam = 0.5
nx = 181
nt = 3000
h = 1/(nx - 1)
u_upwind = solve_equation.upwind(lam, nx, nt)
u_lax = solve_equation.lax(lam, nx, nt)
u_lax_wendroff = solve_equation.lax_wendroff(lam, nx, nt)
u_mac_cormack = solve_equation.mac_cormack(lam, nx, nt)
u_exact = solve_equation.exact_solution(lam, nx, nt)
error_upwind = solve_equation.error(u_upwind, u_exact)
error_lax = solve_equation.error(u_lax, u_exact)
error_lax_wendroff = solve_equation.error(u_lax_wendroff, u_exact)
error_mac_cormack = solve_equation.error(u_mac_cormack, u_exact)

x = np.zeros((nx,1))
for j in range(nx):
    x[j, 0] = j*h

plt.close('all')
plt.plot(x, error_upwind[:, nt], label='Upwind')
plt.plot(x, error_lax[:, nt], label='Lax')
plt.plot(x, error_lax_wendroff[:, nt], label='Lax-Wendroff')
plt.plot(x, error_mac_cormack[:, nt], label='MacCormack', linestyle='--')
plt.xlabel('x')
plt.ylabel('error')
plt.yscale('log')
plt.title('error of numerical solutions')
plt.legend()
plt.savefig("error of numerical solutions.png", dpi=300)
plt.show()

plt.close('all')
plt.plot(x, u_exact[:, nt], label='exact')
plt.plot(x, u_upwind[:, nt], label='Upwind')
plt.plot(x, u_lax[:, nt], label='Lax')
plt.plot(x, u_lax_wendroff[:, nt], label='Lax-Wendroff')
plt.plot(x, u_mac_cormack[:, nt], label='MacCormack', linestyle='--')
plt.xlabel('x')
plt.ylabel('u')
plt.title('u(x,t)')
plt.legend()
plt.savefig("u(x,t).png", dpi=300)
plt.show()
