import pandas as pd
import numpy as np
from sympy import sin, cos, sqrt
from newtonSolver import newtonSolver

def dis(x, y):
    return sqrt(x ** 2 + y ** 2)

def cor_p(alpha):
    return 2 * cos(alpha), 2 * sin(alpha)

def cor_e(beta):
    return 6 + 4 * (cos(beta) ** 2 + sin(beta) * cos(beta)), 4 * (sin(beta) ** 2 + sin(beta) * cos(beta))

def l_2pe_pf(alpha, beta):
    x_f, y_f = 0, 4
    x_p, y_p = cor_p(alpha)
    x_e, y_e = cor_e(beta)
    return dis(x_f - x_p, y_f - y_p) + 2 * dis(x_e - x_p, y_e - y_p)

if __name__ == '__main__':
    di = lambda x, a, b : x + l_2pe_pf(a, b)
    solver = newtonSolver(di, 1, 2)
    df = pd.DataFrame([0])
    solver.solve(df, [0.66, 1.8])
