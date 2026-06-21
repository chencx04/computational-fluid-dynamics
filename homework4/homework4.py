import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.01

# 求解 u_star 和 v_star
def solve_u_star_and_v_star(alpha, num, u, v):
    h = 2 / num
    u_star = np.zeros(num + 1, dtype=complex)
    v_star = np.zeros(num + 1, dtype=complex)
    for j in range(num + 1):
        y = j * h - 1
        u_star[j] = u[j] * (1 - (1 - y**2) * alpha *1j * dt) + 2 * y *v[j] * dt
        v_star[j] = v[j] * (1 - (1 - y**2) * alpha *1j * dt)
    return u_star, v_star

# 求解压强 p 的泊松方程
def solve_pressure_equation(alpha, num, u_star, v_star):
    h = 2 / num
    b = np.zeros(num + 1, dtype=complex)
    b[0] = 1.0
    a_matrix = np.zeros((num + 1, num + 1), dtype=complex)
    a_matrix[0, 0] = 1
    a_matrix[num, num] = 1
    for j in range(num):
        if j != 0:
            a_matrix[j, j] = - alpha**2 - 2 / h**2
            a_matrix[j, j-1] = 1 / h**2
            a_matrix[j, j+1] = 1 / h**2
            b[j] = 1j * alpha * u_star[j] + (v_star[j + 1] - v_star[j - 1]) / 2 / h
    pressure = np.linalg.solve(a_matrix, b)
    return pressure

# 求解 u_starstar 和 v_starstar
def solve_u_starstar_and_v_starstar(alpha, num, u_star, v_star, pressure):
    h = 2 / num
    u_starstar = np.zeros(num + 1, dtype=complex)
    v_starstar = np.zeros(num + 1, dtype=complex)
    for j in range(num + 1):
        y = j * h - 1
        if j !=0 and j != num:
            u_starstar[j] = u_star[j] - 1j * alpha * pressure[j] * dt
            v_starstar[j] = v_star[j] - (pressure[j + 1] - pressure[j - 1]) / 2 / h * dt
        else:
            u_starstar[j] = 0.0
            v_starstar[j] = 0.0
    return u_starstar, v_starstar

# 求解 u_next 和 v_next
def solve_u(alpha, num, u, u_starstar, re):
    h = 2 / num
    u_next = np.zeros(num + 1, dtype=complex)
    a_matrix = np.zeros((num + 1, num + 1), dtype=complex)
    b = np.zeros(num + 1, dtype=complex)
    a_matrix[0, 0] = 1
    a_matrix[num, num] = 1
    for j in range(num):
        if j != 0:
            a_matrix[j, j] = 1 + alpha**2 / 2 / re * dt + 1 / re / h**2 * dt
            a_matrix[j, j-1] = - 1 / 2 / re / h**2 * dt
            a_matrix[j, j+1] = - 1 / 2 / re / h**2 * dt
            b[j] = u_starstar[j] - (alpha**2 + 2 / h**2) / 2 / re * u[j] * dt + (u[j + 1] - u[j - 1]) / 2 / re / h**2 * dt
    u_next = np.linalg.solve(a_matrix, b)
    return u_next

def solve_equation(alpha, num, re, nt):
    # 初值
    u_initial = np.zeros(num + 1, dtype=complex)
    v_initial = np.zeros(num + 1, dtype=complex)
    u = np.zeros((nt + 1, num + 1), dtype=complex)
    v = np.zeros((nt + 1, num + 1), dtype=complex)
    u_next = np.zeros(num + 1, dtype=complex)
    v_next = np.zeros(num + 1, dtype=complex)
    p = np.zeros((nt + 1, num + 1), dtype=complex)
    h = 2 / num
    for j in range(num + 1):
        y = j * h - 1
        u_initial[j] = 1.0e-5 * (1 - y**2) * (1 + y)
    for n in range(nt + 1):
        if n == 0:
            u[n, :] = u_initial
            v[n, :] = v_initial
        else:
            u_star, v_star = solve_u_star_and_v_star(alpha, num, u_initial, v_initial)
            pressure = solve_pressure_equation(alpha, num, u_star, v_star)
            u_starstar, v_starstar = solve_u_starstar_and_v_starstar(alpha, num, u_star, v_star, pressure)
            u_next = solve_u(alpha, num, u_initial, u_starstar, re)
            v_next = solve_u(alpha, num, v_initial, v_starstar, re)
            u[n, :] = u_next
            v[n, :] = v_next
            p[n, :] = pressure
            u_initial = u_next
            v_initial = v_next
    return u, v, p

def get_dissipation(u, v, p, num, nt, x, alpha):
    dissipation_u = np.zeros((nt + 1, num - 1))
    dissipation_v = np.zeros((nt + 1, num - 1))
    dissipation_p = np.zeros((nt + 1, num - 1))
    for n in range(nt + 1):
        for j in range(num - 1):
            dissipation_u[n, j] = abs(u[n, j + 1] * np.exp(1j * alpha * x))
            dissipation_v[n, j] = abs(v[n, j + 1] * np.exp(1j * alpha * x))
            dissipation_p[n, j] = abs(p[n, j + 1] * np.exp(1j * alpha * x))
    return dissipation_u, dissipation_v, dissipation_p

x = 1.0
alpha = 1.0
num = 200
nt = 100
re1 = 5000
re2 = 6000
re3 = 7000

u1, v1, p1 = solve_equation(alpha, num, re1, nt)
u2, v2, p2 = solve_equation(alpha, num, re2, nt)
u3, v3, p3 = solve_equation(alpha, num, re3, nt)
dissipation_u1, dissipation_v1, dissipation_p1 = get_dissipation(u1, v1, p1, num, nt, x, alpha)
dissipation_u2, dissipation_v2, dissipation_p2 = get_dissipation(u2, v2, p2, num, nt, x, alpha)
dissipation_u3, dissipation_v3, dissipation_p3 = get_dissipation(u3, v3, p3, num, nt, x, alpha)

y = np.zeros(num - 1)
for j in range(num - 1):
    y[j] = (j + 1) * (2 / num) - 1

# 绘制动图
fig, ax = plt.subplots()
lines = [
    ax.plot([], [], label='re = 5000')[0],
    ax.plot([], [], label='re = 6000')[0],
    ax.plot([], [], label='re = 7000')[0]
]
ax.set_xlim(-1, 1)
ax.set_ylim(np.min([dissipation_u1.min(), dissipation_u2.min(), dissipation_u3.min()]), 
            np.max([dissipation_u1.max(), dissipation_u2.max(), dissipation_u3.max()]))
ax.set_xlabel('y')
ax.set_ylabel('dissipation')
ax.set_title('dissipation of u')
ax.set_yscale('log')
ax.legend()

def animate(frame):
    for l, data in zip(
        lines, 
        [dissipation_u1, dissipation_u2, dissipation_u3]
    ):
        l.set_data(y, data[frame, :])
    ax.set_title(f'dissipation of u (t = {frame})')
    return lines

ani = animation.FuncAnimation(
    fig, animate, frames=range(0, nt+1, 10), interval=500, blit=False
)
plt.show()
