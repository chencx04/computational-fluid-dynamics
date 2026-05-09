import numpy as np
import matplotlib.pyplot as plt

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

# solve_coefficient()


def plot_error_for_fdm():
    # 定义一个步长 h 的数组
    h_array = np.linspace(0.05,1.05,101)
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
    plt.plot(h_array, f_prime3, label='3-point')
    plt.plot(h_array, f_prime5, label='5-point')
    plt.plot(h_array, f_prime_analytical, label='analytical')
    plt.xlabel('h')
    plt.ylabel('f_prime')
    plt.legend()
    plt.savefig('result.png')

plot_error_for_fdm()

