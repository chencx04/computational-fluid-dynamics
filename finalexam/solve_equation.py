import numpy as np

# 用一阶向前欧拉格式求解守恒形一维欧拉方程
# 已知 n 时刻的解 u_n (包含虚拟点)，求解 n+1 时刻的解 u_n+1
def get_next_state(u_n, flux, lam, num):
    # 得到包含边界外虚拟点的 u_n ，共 num + 5 个点
    u_n_next = np.zeros((num + 6, 3))
    for i in range(num + 1):
        # 需要计算的点为 u_n_next[2] - u_n_next[num+2] 
        j = i + 2
        u_n_next[j] = u_n[j] - lam * (flux[i] - flux[i-1])
    u_n_next[0] = u_n_next[2]
    u_n_next[1] = u_n_next[2]
    u_n_next[num + 3] = u_n_next[num + 2]
    u_n_next[num + 4] = u_n_next[num + 2]
    u_n_next[num + 5] = u_n_next[num + 2]
    return u_n_next

# 子模板内插值公式 p_k(x_j+1/2)
# u1, u2, u3 均为 3 维向量
p0 = lambda u1, u2, u3: 1/3* u1 - 7/6* u2 + 11/6* u3
p1 = lambda u1, u2, u3: -1/6* u1 + 5/6* u2 + 1/3* u3
p2 = lambda u1, u2, u3: 1/3* u1 + 5/6* u2 - 1/6* u3
# 负通量的 f0 = p2(j+1, j, j-1)
# f1 = p1(j+2, j+1, j)
# f2 = p0(j+3, j+2, j+1)

# 光滑指示器
beta0 = lambda u1, u2, u3: 13/12*(u1 - 2*u2 + u3)**2 + 1/4*(u1 - 4*u2 + 3*u3)**2 
beta1 = lambda u1, u2, u3: 13/12*(u1 - 2*u2 + u3)**2 + 1/4*(u1 - u3)**2 
beta2 = lambda u1, u2, u3: 13/12*(u1 - 2*u2 + u3)**2 + 1/4*(3*u1 - 4*u2 + u3)**2 
# 负通量的 beta0 = beta2(j+1, j, j-1)
# beta1 = beta1(j+2, j+1, j)
# beta2 = beta0(j+3, j+2, j+1)

# 理想线性权重
d0 = 1/10
d1 = 3/5
d2 = 3/10
# 负通量的线性权重相反

# 权重 omega
def omega(u1, u2, u3, u4, u5, eps, sig):
    if sig == 1: # 正通量 
        b0 = beta0(u1, u2, u3)
        b1 = beta1(u2, u3, u4)
        b2 = beta2(u3, u4, u5)
    else: # 负通量
        b0 = beta0(u5, u4, u3)
        b1 = beta1(u4, u3, u2)
        b2 = beta2(u3, u2, u1)
    alpha0 = d0/(eps + b0)**2
    alpha1 = d1/(eps + b1)**2
    alpha2 = d2/(eps + b2)**2
    sum_alpha = alpha0 + alpha1 + alpha2
    return alpha0/sum_alpha, alpha1/sum_alpha, alpha2/sum_alpha

# 得到数值正通量的表达式
def get_flux_plus(u1, u2, u3, u4, u5, eps):
    w0, w1, w2 = omega(u1, u2, u3, u4, u5, eps, 1)
    f0 = w0 * p0(u1, u2, u3)
    f1 = w1 * p1(u2, u3, u4)
    f2 = w2 * p2(u3, u4, u5)
    return f0 + f1 + f2

# 得到数值负通量的表达式
def get_flux_minus(u1, u2, u3, u4, u5, eps):
    w0, w1, w2 = omega(u1, u2, u3, u4, u5, eps, 0)
    f0 = w0 * p0(u5, u4, u3)
    f1 = w1 * p1(u4, u3, u2)
    f2 = w2 * p2(u3, u2, u1)
    return f0 + f1 + f2

def get_lambda_star(u, num):
    # u 为 num + 1 个 3 维向量，不包括边界外虚拟点
    # 得到全局最大特征值
    lam_star = np.zeros(num + 1)
    for i in range(num + 1):
        # 在 x_i 处：
        u_curr = u[i]
        # 当前的速度大小
        v = np.abs(u_curr[1] / u_curr[0])
        # 当前的声速大小
        c = np.sqrt(1.4 * (1.4 - 1) * (u_curr[2] / u_curr[0] - 1/2 * v ** 2))
        lam_star[i] = v + c
    return max(lam_star)

# 物理通量的表达式 F(U)
flux_phy = lambda u: np.array([u[1], 0.8 * u[1] ** 2 / u[0] + 0.4 * u[2], 1.4 * u[1] * u[2] / u[0] - 0.2 * u[1] ** 3 / u[0] ** 2])

# 得到数值通量
def get_flux_num(u, num, eps, switch):
    # u 为 num + 6 个 3 维向量，包括边界外虚拟点
    # switch = 1 时，使用 Lax-Friedrichs 格式
    # switch = 2 时，使用 Steger-Warming 格式
    if switch == 1:
        lam_star = get_lambda_star(u[2:num + 3], num)
    u_bar_plus = np.zeros((num + 6, 3)) # 物理正通量在格点处的值
    u_bar_minus = np.zeros((num + 6, 3)) # 物理负通量在格点处的值
    flux_plus = np.zeros((num + 2, 3)) # 数值正通量
    flux_minus = np.zeros((num + 2, 3)) # 数值负通量
    for i in range(num + 1):
	# 得到 x_i+1/2 处的物理通量
        if switch == 1:
            # Lax-Friedrichs
            u_bar_plus[i + 2] = 0.5 * (flux_phy(u[i + 2]) + lam_star * u[i + 2])
            u_bar_minus[i + 2] = 0.5 * (flux_phy(u[i + 2]) - lam_star * u[i + 2])
        else:
	    # Steger-Warming
            l = np.zeros(3, 3) # A 的左特征向量组成的矩阵
            u_curr = u[i + 2] 
            # 当前的速度大小
            v = np.abs(u_curr[1] / u_curr[0])
            # 当前的声速大小
            c = np.sqrt(1.4 * (1.4 - 1) * (u_curr[2] / u_curr[0] - 1/2 * v ** 2))
            lam_matrix = np.array([v - c, 0, 0], [0, v, 0], [0, 0, v + c])
            lam_matrix_abs = np.array([np.abs(v - c), 0, 0], [0, np.abs(v), 0], [0, 0, np.abs(v + c)])
            lam_plus = 0.5 * (lam_matrix + lam_matrix_abs)
            lam_minus = 0.5 * (lam_matrix - lam_matrix_abs)
            l = 0.2 / c ** 2 * np.array([0.5 * v ** 2 + v * c / 0.4, - v - c / 0.4, 1], [-0.5 * v ** 2 + v * c / 0.4, v, -1], [0.5 * v ** 2 - v * c / 0.4, - v + c / 0.4, 1])
            l_inv = np.linalg.inv(l)
            u_bar_plus[i + 2] = l_inv @ lam_plus @ l @ u_curr
            u_bar_minus[i + 2] = l_inv @ lam_minus @ l @ u_curr
    # 边界外虚拟点（Neumann 边界条件）
    u_bar_plus[0] = u_bar_plus[2]
    u_bar_plus[1] = u_bar_plus[2]
    u_bar_plus[num + 5] = u_bar_plus[num + 2]
    u_bar_plus[num + 4] = u_bar_plus[num + 2]
    u_bar_plus[num + 3] = u_bar_plus[num + 2]
    u_bar_minus[0] = u_bar_minus[2]
    u_bar_minus[1] = u_bar_minus[2]
    u_bar_minus[num + 5] = u_bar_minus[num + 2]
    u_bar_minus[num + 4] = u_bar_minus[num + 2]
    u_bar_minus[num + 3] = u_bar_minus[num + 2]
    for i in range(num + 1):
        # x_j+1/2 处的数值通量 
        flux_plus[i] = get_flux_plus(u_bar_plus[i], u_bar_plus[i+1], u_bar_plus[i+2], u_bar_plus[i+3], u_bar_plus[i+4], eps)
        flux_minus[i] = get_flux_minus(u_bar_minus[i+1], u_bar_minus[i+2], u_bar_minus[i+3], u_bar_minus[i+4], u_bar_minus[i+5], eps)
    return flux_plus + flux_minus

num = 11
eps = 1.0e-5
switch = 1
nt = 1
u = np.zeros((num + 6, 3))
u_next = np.zeros((num + 6, 3))
lam = 0.9
for n in range(nt + 1):
    if n == 0:
        for j in range(num + 1):
            if j/num < 0.5:
                u[j + 2] = np.array([1,0,2.5])
            else:
                u[j + 2] = np.array([0.125,0,0.25])
    else:
        flux = get_flux_num(u, num, eps, switch)
        u_next = get_next_state(u, flux, lam, num)
        u = u_next

