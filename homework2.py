import numpy as np

def solve_coefficient():
    a = np.array([
        [1,0,0,0,0],
        [1,1,1,1,1],
        [1,2,4,8,16],
        [1,3,9,27,81],
        [1,4,16,64,256]])

    b = np.array([0,1,0,0,0])

    c = np.linalg.solve(a,b)

    print(c)

solve_coefficient()
