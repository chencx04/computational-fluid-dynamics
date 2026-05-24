import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SolveEquation:
    def upwind(self, lam, nx, nt):
        h = 1 / (nx - 1)
        u = np.zeros((nx, nt+1))
        # 初值
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        for i in range(nt):
            # 已知 i 步的 u ，求解 i+1 步的 u
            for j in range(nx):
                if j != 0:
                    u[j, i + 1] = (1-lam)*u[j, i] + lam*u[j-1, i]
            # 周期边界条件
            u[0, i + 1] = u[nx - 1, i + 1]
        return u
    
    def lax(self, lam, nx, nt):
        h = 1 / (nx - 1)
        u = np.zeros((nx, nt+1))
        # 初值
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        for i in range(nt):
            # 已知 i 步的 u ，求解 i+1 步的 u
            for j in range(nx):
                if j != 0 and j != nx-1:
                    u[j, i + 1] = (1-lam)/2*u[j+1, i]+(1+lam)/2*u[j-1,i]
                elif j == nx-1:
                    # 周期边界条件：u_{nx}=u_{1}
                    u[j, i + 1] = (1-lam)/2*u[1, i]+(1+lam)/2*u[j-1,i]
            # 周期边界条件
            u[0, i + 1] = u[nx-1, i + 1]
        return u
    
    def lax_wendroff(self, lam, nx, nt):
        h = 1 / (nx - 1)
        u = np.zeros((nx, nt+1))
        # 初值
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        for i in range(nt):
            for j in range(nx):
                if j != 0 and j != nx-1:
                    u[j, i + 1] = (1-lam**2)*u[j, i]-(lam/2)*(1-lam)*u[j+1,i]+(lam/2)*(1+lam)*u[j-1,i]
                elif j == nx-1:
                    # 周期边界条件：u_{nx}=u_{1}
                    u[j, i + 1] = (1-lam**2)*u[j, i]-(lam/2)*(1-lam)*u[1,i]+(lam/2)*(1+lam)*u[j-1,i]
            # 周期边界条件
            u[0, i + 1] = u[nx-1, i + 1]
        return u

    def mac_cormack(self, lam, nx, nt):
        h = 1 / (nx - 1)
        u = np.zeros((nx, nt+1))
        # 初值
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
        u_bar = np.zeros((nx, nt+1))
        for i in range(nt):
            # 已知 i 步的 u ，求解 i+1 步的 u_bar
            for j in range(nx):
                if j != nx-1:
                    u_bar[j, i+1] = (1+lam)*u[j,i]-lam*u[j+1,i]
                else:
                    u_bar[j, i+1] = (1+lam)*u[j,i]-lam*u[1,i]
            # 已知 i+1 步的 u_bar ，求解 i+1 步的 u
            for j in range(nx):
                if j !=0 :
                    u[j, i+1] = (u[j,i]+u_bar[j,i+1])/2 - lam*(u_bar[j,i+1]-u_bar[j-1,i+1])/2
            # 周期边界条件
            u[0, i+1] = u[nx-1, i+1]
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
    
    def crank_nicolson(self, lam, nx, nt):
        h = 1 / (nx - 1)
        matrix_a = np.zeros((nx, nx))
        matrix_b = np.zeros((nx, nx))
        u = np.zeros((nx, nt+1))
        b = np.zeros((nx, nt+1))
        # 构建 u(x,t) 的初值，和矩阵 A , B
        for j in range(nx):
            u[j, 0] = np.sin(2*np.pi*j*h)
            if j != 0:
                matrix_a[j, j] = 1 + lam/2
                matrix_a[j, j-1] = -lam/2
                matrix_b[j, j] = 1 - lam/2
                matrix_b[j, j-1] = lam/2
            else:
                matrix_a[j, j] = 1
                matrix_a[j, nx-1] = -1
        # 计算矩阵 C = A^(-1) * B
        matrix_c = np.linalg.inv(matrix_a) @ matrix_b
        for i in range(nt):
            # 已知 i 步的 u ，求解 i+1 步的 u
            u[:, i+1] = matrix_c @ u[:, i]
        return u

class PlotError:
    def compare_explicit(self, lam, nx, nt):
        # 绘制第1题四种显式差分格式的解及其误差
        solve_equation = SolveEquation()
        h = 1/(nx - 1)
        # 横坐标
        x = np.zeros((nx,1))
        for j in range(nx):
            x[j, 0] = j*h
        u_upwind = solve_equation.upwind(lam, nx, nt)
        u_lax = solve_equation.lax(lam, nx, nt)
        u_lax_wendroff = solve_equation.lax_wendroff(lam, nx, nt)
        u_mac_cormack = solve_equation.mac_cormack(lam, nx, nt)
        u_exact = solve_equation.exact_solution(lam, nx, nt)
        error_upwind = solve_equation.error(u_upwind, u_exact)
        error_lax = solve_equation.error(u_lax, u_exact)
        error_lax_wendroff = solve_equation.error(u_lax_wendroff, u_exact)
        error_mac_cormack = solve_equation.error(u_mac_cormack, u_exact)
        # 画图，画出误差图
        plt.close('all')
        plt.plot(x, error_upwind[:, nt], label='Upwind')
        plt.plot(x, error_lax[:, nt], label='Lax')
        plt.plot(x, error_lax_wendroff[:, nt], label='Lax-Wendroff')
        plt.plot(x, error_mac_cormack[:, nt], label='MacCormack', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('error')
        plt.yscale('log')
        plt.title('error of numerical solutions, t = %d' % nt)
        plt.legend()
        plt.savefig("error of explicit numerical solutions with nt = %d.png" % nt, dpi=300)
        # plt.show()

        # 画出 u(x,t) 图
        plt.close('all')
        plt.plot(x, u_exact[:, nt], label='exact')
        plt.plot(x, u_upwind[:, nt], label='Upwind')
        plt.plot(x, u_lax[:, nt], label='Lax')
        plt.plot(x, u_lax_wendroff[:, nt], label='Lax-Wendroff')
        plt.plot(x, u_mac_cormack[:, nt], label='MacCormack', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('u(x,t), t = %d' % nt)
        plt.legend()
        plt.savefig("u(x,t)_explicit with nt = %d.png" % nt, dpi=300)
        # plt.show()

    def compare_implicit(self, lam, nx, nt):
        # 绘制第2题隐式格式的解及其误差
        # 比较迎风格式和Crank-Nicolson格式的数值稳定性及误差大小
        solve_equation = SolveEquation()
        h = 1/(nx - 1)
        # 横坐标
        x = np.zeros((nx,1))
        for j in range(nx):
            x[j, 0] = j*h
        u_crank_nicolson = solve_equation.crank_nicolson(lam, nx, nt)
        u_upwind = solve_equation.upwind(lam, nx, nt)
        u_exact = solve_equation.exact_solution(lam, nx, nt)
        error_crank_nicolson = solve_equation.error(u_crank_nicolson, u_exact)
        error_upwind = solve_equation.error(u_upwind, u_exact)
        plt.close('all')
        plt.plot(x, u_exact[:, nt], label='exact')
        plt.plot(x, u_crank_nicolson[:, nt], label='Crank-Nicolson')
        plt.plot(x, u_upwind[:, nt], label='Upwind')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('u(x,t), t = %d' % nt)
        plt.legend()
        plt.savefig("u(x,t)_crank_nicolson with nt = %d.png" % nt, dpi=300)
        # plt.show()

        plt.close('all')
        plt.plot(x, error_crank_nicolson[:, nt], label='Crank-Nicolson')
        plt.plot(x, error_upwind[:, nt], label='Upwind')
        plt.xlabel('x')
        plt.ylabel('error')
        plt.yscale('log')
        plt.title('error of numerical solutions, t = %d' % nt)
        plt.legend()
        plt.savefig("error of implicit numerical solutions with nt = %d.png" % nt, dpi=300)
        # plt.show()

    def stability(self, lam, nx, nt):
        solve_equation = SolveEquation()
        h = 1/(nx - 1)
        # 横坐标
        x = np.zeros((nx,1))
        for j in range(nx):
            x[j, 0] = j*h
        u_upwind = solve_equation.upwind(lam, nx, nt)
        u_crank_nicolson = solve_equation.crank_nicolson(lam, nx, nt)
        u_exact = solve_equation.exact_solution(lam, nx, nt)
        
        # 画动图
        plt.close('all')
        fig, ax = plt.subplots()

        # 设置坐标轴范围，防止动图跳动
        ax.set_xlim(0, 1)

        # 绘制初始时刻的速度
        ax.plot(x, u_exact[:,0], linewidth = 1.5, color='black')
        ax.set_xlabel('x')
        ax.set_title('u(x,t)')

        line1, = ax.plot([], [], linewidth = 1, label='Crank-Nicolson')
        line2, = ax.plot([], [], '--', linewidth=1, label='Upwind')
        line3, = ax.plot([], [], '--', linewidth=1, label='exact')
        ax.legend(loc='upper right')

        def update_plot(time, x, u1, u2, u3, line1, line2, line3):
            # 更新绘制的数据，形成动图
            line1.set_data(x, u1[:, time])
            line2.set_data(x, u2[:, time])
            line3.set_data(x, u3[:, time])
            # 设置坐标轴范围
            min_y = min(
                min(u1[:, time]),
                min(u2[:, time]),
                -1,
            )
            max_y = max(
                max(u1[:, time]),
                max(u2[:, time]),
                1,
            )
            ax.set_ylim(min_y, max_y)
            return line1, line2, line3,

        ani = FuncAnimation(fig, update_plot, frames=nt, fargs=(x, u_crank_nicolson, u_upwind, u_exact, line1, line2, line3), interval=20, blit=True)
        ani.save('u(x,t)_with_lambda=%.2f.gif' % lam, writer='pillow', fps=30)

plt_err = PlotError()

# 第1题
# 数值耗散误差
lam = 0.5
nx = 101
nt = 300
plt_err.compare_explicit(lam, nx, nt)

# 色散误差
nt = 3000
plt_err.compare_explicit(lam, nx, nt)

# 第2题
# 数值耗散误差
nt = 300
plt_err.compare_implicit(lam, nx, nt)

# 数值稳定性
lam = 1.1
nx = 101
nt = 500
plt_err.stability(lam, nx, nt)


