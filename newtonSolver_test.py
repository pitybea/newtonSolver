from newtonSolver import newtonSolver
from sympy import Symbol
import numpy as np
import pandas as pd

if __name__ == '__main__':

    func = lambda x, a, b, c : a * x * x + b * x + c + b * c * x * x

    dis = lambda x, y, a, b, c : (y - func(x, a, b, c)) ** 2
    
    solver = newtonSolver(dis, 2, 3)
    solver.print_functions()

    x = np.linspace(-5, 10, 150)
    y = func(x, 3, 1, -2)

    df = pd.DataFrame({0: x, 1: y})
    print(solver.solve(df, [1, 1, -3]))
    
