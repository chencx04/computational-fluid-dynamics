import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

class Task1():
    def __init__(self):
        pass

    # 求解差分格式的系数    
    def solve_coefficient():
        a = np.array([
            [1,0,0,0,0],
            [1,1,1,1,1],
            [1,2,4,8,16],
            [1,3,9,27,81],
            [1,4,16,64,256]])

        b = np.array([0,1,0,0,0])

        c = np.linalg.solve(a.T,b)

        print(c)

    """ 计算差分格式的结果，并绘制图像（可选） """
    def calculate_f_prime_with_one_side_nodes_fdm(self, isplot, h_array):
        # 定义一个步长 h 的数组
        # h_array = np.linspace(0.05,1.05,101)
        # 函数 f(x)
        f = lambda x: np.exp(2*x)*2
        # 解析形式的导数 f'(x = -1)
        f_prime_analytical = 4*np.exp(2*(-1)) + np.zeros_like(h_array)
        # 初始化
        f_prime3 = np.zeros_like(h_array)
        f_prime5 = np.zeros_like(h_array)
        for i in range(len(h_array)):
            h = h_array[i]
            # 定义网格点（模板）
            x0 = -1
            x1 = -1 + h
            x2 = -1 + 2*h
            x3 = -1 + 3*h
            x4 = -1 + 4*h
            # 3 节点模板的有限差分格式
            f_prime3[i] = (-3*f(x0) + 4*f(x1) -f(x2))/(2*h)
            # 5 节点模板的有限差分格式
            f_prime5[i] = (-25*f(x0) + 48*f(x1) - 36*f(x2) + 16*f(x3) - 3*f(x4))/(12*h)

        # plot error-h figure
        if isplot == 1:
            plt.close('all')
            plt.plot(h_array, f_prime3, label='3-point')
            plt.plot(h_array, f_prime5, label='5-point')
            plt.plot(h_array, f_prime_analytical, label='analytical')
            plt.xlabel('h')
            plt.ylabel('f_prime')
            plt.legend()
            plt.title('x = -1')
            plt.savefig('result_x=-1.png')
        elif isplot == 2:
            plt.close('all')
            plt.plot(h_array, abs(f_prime3 - f_prime_analytical), label='3-point')
            plt.plot(h_array, abs(f_prime5 - f_prime_analytical), label='5-point')
            plt.xlabel('h')
            plt.ylabel('error')
            plt.title('x = -1')
            plt.legend()
            plt.savefig('error_x=-1.png')


        return h_array, f_prime3, f_prime5, f_prime_analytical


    def calculate_f_prime_with_symmetric_nodes_fdm(self, isplot, h_array):
        # 定义一个步长 h 的数组
        # h_array = np.linspace(0.05,1.05,101)
        # 函数 f(x)
        f = lambda x: np.exp(2*x)*2
        # 解析形式的导数 f'(x = 0)
        f_prime_analytical = 4*np.exp(2*0) + np.zeros_like(h_array)
        # 初始化
        f_prime3 = np.zeros_like(h_array)
        f_prime5 = np.zeros_like(h_array)
        for i in range(len(h_array)):
            h = h_array[i]
            # 定义网格点（模板）
            x0 = 0
            x1 = h
            x2 = 2*h
            x_1 = -h # x_{-1}
            x_2 = -2*h # x_{-2}
            # 3 节点模板的有限差分格式
            f_prime3[i] = (f(x1) -f(x_1))/(2*h)
            # 5 节点模板的有限差分格式
            f_prime5[i] = (8*f(x1) - f(x2) - 8*f(x_1) + f(x_2))/(12*h)
        
        # plot error-h figure
        plt.close('all')
        if isplot == 1:
            plt.plot(h_array, f_prime3, label='3-point')
            plt.plot(h_array, f_prime5, label='5-point')
            plt.plot(h_array, f_prime_analytical, label='analytical')
            plt.xlabel('h')
            plt.ylabel('f_prime')
            plt.title('x = 0')
            plt.legend()
            plt.savefig('result_x=0.png')
        elif isplot == 2:
            plt.plot(h_array, abs(f_prime3 - f_prime_analytical), label='3-point')
            plt.plot(h_array, abs(f_prime5 - f_prime_analytical), label='5-point')
            plt.xlabel('h')
            plt.ylabel('error')
            plt.title('x = 0')
            plt.legend()
            plt.savefig('error_x=0.png')


        return h_array, f_prime3, f_prime5, f_prime_analytical

        

    """ 绘制差分格式误差图，以计算误差阶数 """
    def calculate_order_of_accuracy_for_fdm(self, h_array, x0):
        if x0 == -1:
            _, f_prime3, f_prime5, f_prime_analytical = self.calculate_f_prime_with_one_side_nodes_fdm(isplot=0, h_array=h_array)
        elif x0 == 0:
            _, f_prime3, f_prime5, f_prime_analytical = self.calculate_f_prime_with_symmetric_nodes_fdm(isplot=0, h_array=h_array)
        # 绘制 log-log 图，以确定误差阶数
        log_h_array = np.log(h_array)
        log_error_prime3 = np.log(abs(f_prime3 - f_prime_analytical))
        log_error_prime5 = np.log(abs(f_prime5 - f_prime_analytical))
        # 拟合
        slope3, intercept3, r_value3, _, _ = stats.linregress(log_h_array, log_error_prime3)
        slope5, intercept5, r_value5, _, _ = stats.linregress(log_h_array, log_error_prime5)

        # 画图以可视化
        plt.close('all')
        plt.plot(log_h_array, log_error_prime3, label='3-point, r = %.4f, slope = %.4f' % (r_value3,slope3))
        plt.plot(log_h_array, log_error_prime5, label='5-point, r = %.4f, slope = %.4f' % (r_value5,slope5))
        plt.xlabel('log(h)')
        plt.ylabel('log(error)')
        plt.title('x0 = %.1f, h = %.4f to %.4f' % (x0, h_array[0], h_array[len(h_array)-1]))
        plt.legend()
        plt.savefig('order_of_accuracy_x0 = %.1f %.4f %.4f.png' % (x0, h_array[0], h_array[len(h_array)-1]))

    # 比较 x = -1 和 x = 0 的差分格式的误差大小
    def compare_fdm_error_with_different_x0(self, h_array):
            _, f_prime3_1, f_prime5_1, f_prime_analytical_1 = self.calculate_f_prime_with_one_side_nodes_fdm(isplot=0, h_array=h_array)
            _, f_prime3_0, f_prime5_0, f_prime_analytical_0 = self.calculate_f_prime_with_symmetric_nodes_fdm(isplot=0, h_array=h_array)
            plt.close('all')
            plt.plot(h_array, abs(f_prime3_1 - f_prime_analytical_1), label='3-point, x = -1')
            plt.plot(h_array, abs(f_prime5_1 - f_prime_analytical_1), label='5-point, x = -1')
            plt.plot(h_array, abs(f_prime3_0 - f_prime_analytical_0), label='3-point, x = 0')
            plt.plot(h_array, abs(f_prime5_0 - f_prime_analytical_0), label='5-point, x = 0')
            plt.xlabel('h')
            plt.ylabel('error')
            plt.yscale('log')
            plt.title('error of f_prime with different x0')
            plt.legend()


 
t1 = Task1()

def run_file(i, x_0):
    h_array = np.linspace(0.005,1.005,1001)
    if i == 0:
        t1.solve_coefficient()
        # 计算 5 节点模板的差分格式的系数
    elif i == 1 and x_0 == -1:
        t1.calculate_f_prime_with_one_side_nodes_fdm(isplot=1, h_array=h_array)
        t1.calculate_f_prime_with_one_side_nodes_fdm(isplot=2, h_array=h_array)
        # 绘制 f'(x = -1) 及与解析值的误差随 h 变化的图像
        # 包括 3 节点模板和 5 节点模板的有限差分格式，以及解析形式的导数
    elif i == 1 and x_0 == 0:
        t1.calculate_f_prime_with_symmetric_nodes_fdm(isplot=1, h_array=h_array)
        t1.calculate_f_prime_with_symmetric_nodes_fdm(isplot=2, h_array=h_array)
        # 绘制 f'(x = 0) 及与解析值的误差随 h 变化的图像
        # 包括 3 节点模板和 5 节点模板的有限差分格式，以及解析形式的导数
    elif i == 2:
        t1.compare_fdm_error_with_different_x0(h_array=h_array)
        # 比较 x = -1 和 x = 0 的差分格式的误差
    elif i == 3:
        h_array = np.linspace(0.05,1.05,101)
        t1.calculate_order_of_accuracy_for_fdm(h_array = h_array, x0 = x_0)
        # 绘制双对数图，以计算 f'(x) 的误差阶数，选取步长范围为 0.05 到 1.05，步长为 0.01
    elif i == 4:
        h_array = np.linspace(0.01,0.51,101)
        t1.calculate_order_of_accuracy_for_fdm(h_array = h_array, x0 = x_0)
        # 绘制双对数图，以计算 f'(x) 的误差阶数，选取步长范围为 0.01 到 0.51，步长为 0.005
    elif i == 5:
        h_array = np.linspace(0.001,0.101,101)
        t1.calculate_order_of_accuracy_for_fdm(h_array = h_array, x0 = x_0)
        # 绘制双对数图，以计算 f'(x) 的误差阶数，选取步长范围为 0.001 到 0.101，步长为 0.01
    elif i == 6:
        h_array = np.linspace(0.0001,0.0101,101)
        t1.calculate_order_of_accuracy_for_fdm(h_array = h_array, x0 = x_0)
        # 绘制双对数图，以计算 f'(x) 的误差阶数，选取步长范围为 0.0001 到 0.0101，步长为 0.01

# 运行得到第1题的所有结果
# for i in range(7):
#     if i == 0:
#         pass
#     else:
#         for x_0 in [-1, 0]:
#             run_file(i=i, x_0=x_0)

# 运行得到特定的结果
# run_file(i=2,x_0=0)


class Task3():
    def __init__(self):
        pass

    def solve_equation(self, m, l, nt):
        # m+1 为 x 方向的网格数，l 为网格比 λ, nt 为时间步数
        # x
        x = np.linspace(0,2*np.pi,m+1)
        # Au = b
        # 构造系数矩阵 A
        a = np.zeros((m+1,m+1))
        for j in range(m+1):
            if j == 0:
                a[j,j] = 1
                a[j,m] = -1
            else:
                a[j,j]=1-l
                a[j,j-1]=l
        
        # n=0 时的 b
        b = np.sin(x)
        b[0] = 0

        # 初始化自变量 u
        u = np.zeros((nt,m+1))
        # 求解 1000 个时间步长
        for i in range(nt):
            # 求解 u[i,:]
            if i == 0:
                u[i,:] = np.sin(x)
            else:
                b = u[i-1,:]
                b[0] = 0
                u[i,:] = np.linalg.solve(a,b)

        # 画动态图
        plt.close('all')
        fig, ax = plt.subplots()
        
        # 设置坐标轴范围，防止动图跳动
        ax.set_xlim(0, 2 * np.pi)
        # ax.set_ylim(-1, 1)

        # 绘制初始时刻的速度
        ax.plot(x, u[0,:], linewidth = 1.5, color='black', label='t=0')
        ax.legend()

        line, = ax.plot([], [], linewidth = 1)


        def update_plot(time, u, line):
            # 更新绘制的数据，形成动图
            print('time = %d' % time)
            line.set_ydata(u[time,:])
            return line,

        ani = FuncAnimation(fig, update_plot, frames=nt, fargs=(u, line), interval=20, blit=True)
        ani.save('u(t)_with_lambda=%.2f.gif' % l, writer='pillow', fps=30)



t3 = Task3()
# 步长 h =2*pi/m
t3.solve_equation(m=180, l=1.1, nt=10)


