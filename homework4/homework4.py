import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.01

# 构建矩阵 A = D_h**2 - alpha**2 * I
def a_matrix(alpha, num):
    h = 2 / num
    a_matrix = np.zeros((num + 1, num + 1))
    a_matrix[0, 0] = 1
    a_matrix[num, num] = 1
    for j in range(num):
        if j != 0:
            a_matrix[j, j] = - alpha**2 - 2 / h**2
            a_matrix[j, j-1] = 1 / h**2
            a_matrix[j, j+1] = 1 / h**2
    return a_matrix

# 构建矩阵 B = 1 - 0.5 * L_ss
# L_starstar = L_ss = -i * alpha * (1 - y**2) - 2 * i * alpha * inv(A) + A / Re
def l_ss_matrix(alpha, num, re, a_matrix):
    h = 2 / num
    l_ss_matrix = -2j * alpha * np.linalg.inv(a_matrix) + a_matrix / re
    for j in range(num + 1):
        y = j * h - 1
        l_ss_matrix[j, j] += - 1j * alpha * (1 - y**2)
    return l_ss_matrix * dt

def b_matrix(num, l_ss_matrix):
    b_matrix = 1 - 0.5 * l_ss_matrix
    b_matrix[0, :] = 0.0
    b_matrix[num, :] = 0.0
    b_matrix[0, 0] = 1.0
    b_matrix[num, num] = 1.0
    return b_matrix

def b_vector(num, v_n, l_ss_matrix):
    b_vector = (1 + 0.5 * l_ss_matrix) @ v_n
    b_vector[0] = 0
    b_vector[num] = 0
    return b_vector

def solve_v(alpha, num, re, nt, v_initial):
    v = np.zeros((nt + 1, num + 1), dtype=complex)
    a = a_matrix(alpha, num)
    l_ss = l_ss_matrix(alpha, num, re, a)
    b = b_matrix(num, l_ss)
    # 已知 n - 1 步的 v，求解 n 步的 v
    for n in range(nt + 1):
        if n == 0:
            v[n, :] = v_initial
        else:
            b_vec = b_vector(num, v[n-1, :], l_ss)
            v[n, :] = np.linalg.solve(b, b_vec)
    return v

# u = i / alpha * dv/dy
# 从 v 得到 u
def solve_u(alpha, num, nt, v):
    h = 2 / num
    u = np.zeros((nt + 1, num + 1), dtype=complex)
    for n in range(nt + 1):
        for j in range(num + 1):
            if j != 0 and j != num:
                u[n, j] = 1j / alpha * (v[n, j + 1] - v[n, j - 1]) / 2 / h
            else:
                u[n, j] = 0 # 不滑移
    return u

def solve_equation(alpha, num, re, nt, v_initial):
    v = solve_v(alpha, num, re, nt, v_initial)
    u = solve_u(alpha, num, nt, v)
    return u, v

# 从 \hat{u}, \hat{v} 得到 u, v
def get_disturbance(u, v, num, nt, x, alpha):
    disturbance_u = np.zeros((nt + 1, num + 1))
    disturbance_v = np.zeros((nt + 1, num + 1))
    for n in range(nt + 1):
        disturbance_u[n, :] = abs(u[n, :] * np.exp(1j * alpha * x))
        disturbance_v[n, :] = abs(v[n, :] * np.exp(1j * alpha * x))
    return disturbance_u, disturbance_v

x = 1.0
alpha = 1.0
num = 101
nt = 100000
re1 = 5000
re2 = 6000
re3 = 7000

v_initial = np.zeros(num + 1, dtype=complex)
for j in range(num + 1):
    y = j * (2.0 / num) - 1
    v_initial[j] = 1.0e-5 * (1 - y**2) * (1 + y)

u1, v1 = solve_equation(alpha, num, re1, nt, v_initial)
u2, v2 = solve_equation(alpha, num, re2, nt, v_initial)
u3, v3 = solve_equation(alpha, num, re3, nt, v_initial)
disturbance_u1, disturbance_v1 = get_disturbance(u1, v1, num, nt, x, alpha)
disturbance_u2, disturbance_v2 = get_disturbance(u2, v2, num, nt, x, alpha)
disturbance_u3, disturbance_v3 = get_disturbance(u3, v3, num, nt, x, alpha)

# 处理数据，避免对数坐标报错
def process_data(data):
    min_data = np.min(data)
    data = np.where(data == 0, max(1e-10 * min_data, 1e-30), data)
    return data

disturbance_u1 = process_data(disturbance_u1)
disturbance_v1 = process_data(disturbance_v1)
disturbance_u2 = process_data(disturbance_u2)
disturbance_v2 = process_data(disturbance_v2)
disturbance_u3 = process_data(disturbance_u3)
disturbance_v3 = process_data(disturbance_v3)

y = np.zeros(num + 1)
for j in range(num + 1):
    y[j] = j * (2 / num) - 1

# 绘制动图
def plot_disturbance(y, data1, data2, data3, title):
    fig, ax = plt.subplots()
    lines = [
        ax.plot([], [], label='re = 5000')[0],
        ax.plot([], [], label='re = 6000')[0],
        ax.plot([], [], label='re = 7000')[0]
    ]
    ax.set_xlim(-1, 1)
    ax.set_ylim(np.min([data1.min(), data2.min(), data3.min()]), 
                np.max([data1.max(), data2.max(), data3.max()]))
    ax.set_xlabel('y')
    ax.set_ylabel('disturbance')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()

    def animate(frame):
        for l, data in zip(
            lines, 
            [data1, data2, data3]
        ):
            l.set_data(y, data[frame, :])
        ax.set_title(f'{title} (t = {frame * dt:.2f})')
        return lines

    ani = animation.FuncAnimation(
        fig, animate, frames=range(0, nt+1, 50), interval=20, blit=False
    )
    ani.save(f'{title}.gif', writer='pillow', fps=50)

# 速度扰动的幅值随时间变化
def plot_disturbance_amplitude(u, v, num, nt, re):
    h = 2 / num
    t = np.zeros(nt + 1)
    disturbance_amplitude = np.zeros(nt + 1)
    for n in range(nt + 1):
        disturbance_amplitude[n] = np.sqrt(np.sum(abs(u[n, :])**2 + abs(v[n, :])**2)) * h
        t[n] = n * dt
    return t, disturbance_amplitude



plot_disturbance(y, disturbance_u1, disturbance_u2, disturbance_u3, 'disturbance of u')
plot_disturbance(y, disturbance_v1, disturbance_v2, disturbance_v3, 'disturbance of v')

t, disturbance_amplitude1 = plot_disturbance_amplitude(u1, v1, num, nt, re1)
_, disturbance_amplitude2 = plot_disturbance_amplitude(u2, v2, num, nt, re2)
_, disturbance_amplitude3 = plot_disturbance_amplitude(u3, v3, num, nt, re3)
plt.close('all')
plt.plot(t, disturbance_amplitude1, label=f'Re = {re1}')
plt.plot(t, disturbance_amplitude2, label=f'Re = {re2}')
plt.plot(t, disturbance_amplitude3, label=f'Re = {re3}')
plt.xlabel('t')
plt.ylabel('disturbance amplitude')
plt.legend()
plt.yscale('log')
plt.title('disturbance amplitude vs time')
plt.savefig('disturbance_amplitude_vs_time.png')
plt.show()
