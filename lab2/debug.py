# %%

import numpy as np
import pandas as pd
import pylab


def f_first(X):
    return 100 * (X[1] - X[0] ** 2) ** 2 + (1 - X[0]) ** 2


dfirst = [
    lambda X: 2 * (200 * X[0] ** 3 - 200 * X[0] * X[1] + X[0] - 1),
    lambda X: 200 * (X[1] - X[0] ** 2)
]


def df_first(X, i):
    return dfirst[i](X)


def f_second(X):
    return (X[1] - X[0] ** 2) ** 2 + (1 - X[0]) ** 2


dsecond = [
    lambda X: 2 * (2 * X[0] ** 3 - 2 * X[0] * X[1] + X[0] - 1),
    lambda X: 2 * (X[1] - X[0] ** 2)
]


def df_second(X, i):
    return dsecond[i](X)


def f_third(X):
    return (1.5 - X[0] * (1 - X[1])) ** 2 + (2.25 - X[0] * (1 - X[1] ** 2)) ** 2 + (2.625 - X[0] * (1 - X[1] ** 3)) ** 2


dthird = [
    lambda X: 2 * X[0] * (X[1] ** 6 + X[1] ** 4 - 2 * X[1] ** 3 - X[1] ** 2 - 2 * X[1] + 3) + 5.25 * X[1] ** 3 + 4.5 *
              X[1] ** 2 + 3 * X[1] - 12.75,
    lambda X: X[0] * (X[0] * (6 * X[1] ** 5 + 4 * X[1] ** 3 - 6 * X[1] ** 2 - 2 * X[1] - 2) + 15.75 * X[1] ** 2 + 9 * X[
        1] + 3)
]


def df_third(X, i):
    return dthird[i](X)


def f_fourth(X):
    return (X[0] + X[1]) ** 2 + 5 * (X[2] - X[3]) ** 2 + (X[1] - 2 * X[2]) ** 4 + 10 * (X[0] - X[3]) ** 4


dfourth = [
    lambda X: 2 * (X[0] + X[1]) + 40 * (X[0] - X[3]) ** 3,
    lambda X: 2 * (X[0] + X[1]) + 4 * (X[1] - 2 * X[2]) ** 3,
    lambda X: 10 * (X[2] - X[3]) - 6 * (X[1] - 2 * X[2]) ** 3,
    lambda X: -10 * (X[2] - X[3]) - 40 * (X[0] - X[3]) ** 3
]


def df_fourth(X, i):
    return dfourth[i](X)


np.random.seed(42)

equations = {
    '100(x_2−x_1^2)^2+(1-x_1)^2': [f_first, df_first, np.random.random(len(dfirst)) * 0.1],
    '(x_2−x_1^2)^2+(1-x_1)^2': [f_second, df_second, np.random.random(len(dsecond)) * 0.1],
    '(1.5-x_1*(1-x_2))^2+(2.25-x_1*(1-x_2^2))^2+(2.625-x_1*(1 -x_2^3))^2': [f_third, df_third,
                                                                            np.random.random(len(dthird)) * 0.1],
    '(x_1+x_2)^2+5(x_3−x_4)^2+(x_2−2x_3)^4+10(x_1−x_4)^4': [f_fourth, df_fourth, np.random.random(len(dfourth)) * 0.1]
}


def secant_method(x1, x2, df_x1, df_x2):
    a = (df_x2 - df_x1) / (x2 - x1)
    b = df_x1 - x1 * a
    return -b / a


def brent(f, df, start, end, eps=1e-5):
    u = x = w = v = 0.5 * (start + end)
    f_u = f_x = f_w = f_v = f(x)
    df_u = df_x = df_w = df_v = df(x)
    d = e = end - start

    function_calls_number = 1
    step_details = []

    while True:
        step_details.append([start, end, x, f_x, df_x, w, f_w, df_w, v, f_v, df_v, d, e, u, f_u, df_u])
        g, e = e, d
        candidates = []

        if x != w and df_x != df_w:
            u1 = secant_method(x, w, df_x, df_w)
            if start + eps <= u1 <= end - eps and np.abs(u1 - x) < 0.5 * g:
                candidates.append(u1)

        if x != v and df_x != df_v:
            u2 = secant_method(x, v, df_x, df_v)
            if start + eps <= u2 <= end - eps and np.abs(u2 - x) < 0.5 * g:
                candidates.append(u2)

        if len(candidates) == 2:
            u = candidates[0]
        else:
            if df_x > 0:
                u = 0.5 * (start + x)
            else:
                u = 0.5 * (x + end)

        if np.abs(u - x) < eps:
            u = x + np.sign(u - x) * eps

        d = np.abs(x - u)
        f_u = f(u)
        df_u = df(u)
        function_calls_number += 1
        if f_u <= f_x:
            if u >= x:
                start = x
            else:
                end = x
            v, w, x = w, x, u
            f_v, f_w, f_x = f_w, f_x, f_u
            df_v, df_w, df_x = df_w, df_x, df_u
        else:
            if u >= x:
                end = u
            else:
                start = u
            if f_u <= f_w or w == x:
                v, w = w, u
                f_v, f_w = f_w, f_u
                df_v, df_w = df_w, df_u
            elif f_u <= f_v or v == x or v == w:
                v = u
                f_v = f_u
                df_v = df_u

        if np.abs(u - step_details[-1][-3]) < eps:
            break

    step_details.append([start, end, x, f_x, df_x, w, f_w, df_w, v, f_v, df_v, d, e, u, f_u, df_u])
    return x, function_calls_number, pd.DataFrame(step_details,
                                                  columns=['start', 'end', 'x', 'f_x', 'df_x', 'w', 'f_w', 'df_w', 'v',
                                                           'f_v', 'df_v', 'd', 'e', 'u', 'f_u', 'df_u'])


def golden_ratio(function, start, end, epsilon=1e-5):
    phi = 0.5 * (3 - np.sqrt(5))

    x1 = start + (end - start) * phi
    x2 = end - (end - start) * phi

    x1_value = function(x1)
    x2_value = function(x2)

    while end - start > epsilon:
        if x1_value > x2_value:
            start = x1
            x1 = x2
            x2 = end + start - x1
            x1_value = x2_value
            x2_value = function(x2)
        else:
            end = x2
            x2 = x1
            x1 = start + end - x2
            x2_value = x1_value
            x1_value = function(x1)

    return 0.5 * (x1 + x2)


def coordinate_descent(X0, f, df, eps=1e-5, max_iter=5e4):
    coords = []
    values = []
    args_count = len(X0)
    k = 0

    X = X0.copy()
    coords.append(X)
    values.append(f(X))

    while k < max_iter:
        i = k % args_count
        g = np.zeros(shape=(args_count,), dtype=float)
        g[i] = df(X, i)
        f_next_X = lambda a: X - a * g
        alpha = golden_ratio(lambda a: f(f_next_X(a)), eps, 1, epsilon=eps)
        X = f_next_X(alpha)

        coords.append(X)
        values.append(f(X))
        k += 1

        if np.sum(np.abs(values[-1] - values[-2])) < eps:
            break

    return np.array(coords), np.array(values)


def steepest_descent(X0, f, df, eps=1e-5, max_iter=5e4):
    coords = []
    values = []
    args_count = len(X0)
    k = 0

    X = X0.copy()
    coords.append(X)
    values.append(f(X))

    while k < max_iter:
        g = np.array([df(X, i) for i in range(args_count)])
        f_next_X = lambda a: X - a * g
        alpha = golden_ratio(lambda a: f(f_next_X(a)), eps, 1, epsilon=eps)
        X = f_next_X(alpha)

        coords.append(X)
        values.append(f(X))
        k += 1

        if np.sum(np.abs(values[-1] - values[-2])) < eps:
            break

    return np.array(coords), np.array(values)


def risovalka(coords, f):
    min_x = np.min(coords[:, 0])
    max_x = np.max(coords[:, 0])
    min_y = np.min(coords[:, 1])
    max_y = np.max(coords[:, 1])
    x_size = max_x - min_x
    y_size = max_y - min_y

    y_space = np.linspace(min_y - 0.05 * y_size, max_y + 0.05 * y_size, 1000)
    x_space = np.linspace(min_x - 0.05 * x_size, max_x + 0.05 * x_size, 1000)

    xgrid, ygrid = np.meshgrid(x_space, y_space)
    zgrid = np.zeros(xgrid.shape)
    for i in range(len(xgrid)):
        for j in range(len(xgrid[0])):
            zgrid[i, j] = f(np.array([xgrid[i, j], ygrid[i, j]]))

    pylab.figure(figsize=(8, 8))

    pylab.plot(coords[:, 0], coords[:, 1], color='red')

    pylab.scatter(x=coords[0][0], y=coords[0][1], color='orange', s=50)
    pylab.scatter(x=coords[-1][0], y=coords[-1][1], color='blue', s=50)

    cs = pylab.contour(xgrid, ygrid, zgrid, 21)
    pylab.clabel(cs)
    pylab.show()


gradient_descents = {
    'coordinate': coordinate_descent,
    'steepest': steepest_descent,
}

for descent in gradient_descents:
    for equation in equations:
        f, df, X0 = equations[equation]
        coords, values = gradient_descents[descent](X0, f, df)
        print(f'{descent}, min of {equation} is {coords[-1]} with value {values[-1]}')
        if len(X0) == 2:
            risovalka(coords, f)

unimodal_functions = {
    '-5x^5 + 4x^4 - 12x^3 + 11x^2 - 2x + 1, [-0.5,0.5]': (
        lambda x: -5 * x ** 5 + 4 * x ** 4 - 12 * x ** 3 + 11 * x ** 2 - 2 * x + 1,
        lambda x: -25 * x ** 4 + 16 * x ** 3 - 36 * x ** 2 + 22 * x - 2,
        (-0.5, 0.5)
    ),
    'lg(x-2)^2 + lg(10-x)^2 - x^0.2, [6.9,9]': (
        lambda x: np.log10(x - 2) ** 2 + np.log10(10 - x) ** 2 - np.power(x, 0.2),
        lambda x: -0.2 * np.power(x, -0.8) + 2 * np.log(x - 2) / np.log(10) ** 2 / (x - 2) - 2 * np.log(10 - x) / np.log(
            10) ** 2 / (10 - x),
        (6, 9.9)
    ),
    '-3x*sin(0.75x) + e^(-2x), [0,2pi]': (
        lambda x: -3 * x * np.sin(0.75 * x) + np.exp(-2 * x),
        lambda x: -2 * np.exp(-2 * x) - 2.25 * x * np.cos(0.75 * x) - 3 * np.sin(0.75 * x),
        (0, 2 * np.pi)
    ),
    'e^(3x) + 5e^(-2x), [0,1]': (
        lambda x: np.exp(3 * x) + 5 * np.exp(-2 * x),
        lambda x: 3 * np.exp(3 * x) - 10 * np.exp(-2 * x),
        (0, 1)
    ),
    '0.2x*lg(x) + (x-2.3)^2, [0.5,2.5]': (
        lambda x: 0.2 * x * np.log10(x) + (x - 2.3) ** 2,
        lambda x: np.log(x) / 5 / np.log(10) + 2 * (x - 2.3) + 1 / 5 / np.log(10),
        (0.5, 2.5)
    )
}

for readable_func, func_number in zip(unimodal_functions, range(len(unimodal_functions))):
    func = unimodal_functions[readable_func][0]
    df = unimodal_functions[readable_func][1]
    start_range, end_range = unimodal_functions[readable_func][2]

    data = []
    value, function_calls, step_details = brent(func, df, start_range, end_range)
    data.append([value, func(value), function_calls, len(step_details)])

    print(readable_func)
    print(pd.DataFrame(data, columns=['x', 'y', 'function calls', 'iterations']))
    print()
