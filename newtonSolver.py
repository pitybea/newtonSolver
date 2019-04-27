import pandas as pd
import numpy as np
import sympy
from sympy import diff, Symbol, Matrix
from sympy.utilities.lambdify import lambdify


class newtonSolver:
    def __init__(self, function, variable_num, parameter_num):
        self.function = function
        self.variable_num = variable_num
        self.parameter_num = parameter_num
        variables = [Symbol('v%d' % i) for i in range(self.variable_num)]
        parameters = [Symbol('p%d' % i) for i in range(self.parameter_num)]
        gradients = [diff(function(*(variables + parameters)), p) for p in parameters]
        hessians = [[diff(g, p) for p in parameters] for g in gradients]
        self.gradient_functions = [lambdify(variables + parameters, g) for g in gradients]
        self.hessian_functions = [[lambdify(variables + parameters, h) for h in hs] for hs in hessians]
        self.variables = variables
        self.parameters = parameters

    def print_functions(self):
        inp = self.variables + self.parameters
        print(self.function(*inp))
        print([f(*inp) for f in self.gradient_functions])
        print([[h(*inp) for h in hs] for hs in self.hessian_functions])
        
    def solve(self, df, init_parameters, tolerence = 1e-9, iter_rounds = 500):
        assert type(df) is pd.DataFrame
        assert df.shape[1] == len(self.variables)
        assert len(init_parameters) == len(self.parameters)
        parameters = init_parameters
        round_num = 1
        old_error = 0
        while True:
            error = df.apply(lambda x : self.function(*(list(x[:]) + parameters)), axis = 1).mean()
            if round_num > iter_rounds or (round_num > 1 and np.abs(old_error - error) < tolerence):
                break
            gradients = [df.apply(lambda x : g(*(list(x[:]) + parameters)), axis = 1).mean() for g in self.gradient_functions]
            hessians = [[df.apply(lambda x: h(*(list(x[:]) + parameters)), axis = 1).mean() for h in hs] for hs in self.hessian_functions]
            step = np.array(Matrix(hessians) ** -1 * Matrix(gradients))
            for j in range(len(step)):
                parameters[j] -= float(step[j])
            print(round_num, ': ', parameters)
            old_error = error
            round_num = round_num + 1
        return parameters
