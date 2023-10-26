﻿import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import random


def test_data(k: float = 1.0, b: float = 0.1, arg_range: float = 1.0,
              rand_range: float = 0.05, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерируюет линию вида y = k*x + b + dy, где dy - аддитивный шум с амплитудой half_disp
    :param k: наклон линии
    :param b: смещение по y
    :param arg_range: диапазон аргумента от 0 до arg_range
    :param rand_range: диапазон шума данных
    :param n_points: количество точек
    :return: кортеж значенией по x и y
    """
    x_step = arg_range / (n_points - 1)
    return np.array([i * x_step for i in range(n_points)]), \
           np.array([np.exp(-(i * x_step * k) ** 2) + b +
                     random.uniform(-rand_range * 0.5, rand_range * 0.5) for i in range(n_points)])


def test_data_2d_poly(surf_params: Tuple[float, float, float, float, float, float] = (1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
                      args_range: float = 1.0, rand_range: float = .1, n_points: int = 1000) -> \
                      Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x = np.array([random.uniform(-args_range*0.5, args_range*0.5) for _ in range(n_points)])
    y = np.array([random.uniform(-args_range*0.5, args_range*0.5) for _ in range(n_points)])
    dz = np.array([surf_params[5] + random.uniform(-rand_range * 0.5, rand_range * 0.5) for _ in range(n_points)])
    return x, y, surf_params[0] * x * x + surf_params[1] * y * x + \
                 surf_params[2] * y * y + surf_params[3] * x + surf_params[4] * y + dz


def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, args_range: float = 1.0,
                 rand_range: float = 1.0, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум в диапазоне rand_range
    :param kx: наклон плоскости по x
    :param ky: наклон плоскости по y
    :param b: смещение по z
    :param args_range: диапазон аргументов по кажой из осей от 0 до args_range
    :param rand_range: диапазон шума данных
    :param n_points: количество точек
    :returns: кортеж значенией по x, y и z
    """
    x = np.array([random.uniform(-args_range*0.5, args_range*0.5) for _ in range(n_points)])
    y = np.array([random.uniform(-args_range*0.5, args_range*0.5) for _ in range(n_points)])
    dz = np.array([b + random.uniform(-rand_range * 0.5, rand_range * 0.5) for _ in range(n_points)])
    return x, y, x * kx + y * ky + dz


def test_data_nd(surf_settings: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 12.0]), args_range: float = 1.0,
                 rand_range: float = 0.1, n_points: int = 125) -> np.ndarray:
    """
    Генерирует плоскость вида z = k_0*x_0 + k_1*x_1...,k_n*x_n + d + dz, где dz - аддитивный шум в диапазоне rand_range
    :param surf_settings: параметры плоскости в виде k_0,k_1,...,k_n,d
    :param args_range: диапазон аргументов по кажой из осей от 0 до args_range
    :param n_points: количество точек
    :param rand_range: диапазон шума данных
    :returns: массив из строк вида x_0, x_1,...,x_n, f(x_0, x_1,...,x_n)
    """
    # surf_settings = [nx,ny,nz,d]
    n_dims = surf_settings.size - 1

    data = np.zeros((n_points, n_dims + 1,), dtype=float)

    for i in range(n_dims):
        data[:, i] = np.array([random.uniform(-args_range*0.5, args_range*0.5) for _ in range(n_points)])
        data[:, n_dims] += surf_settings[i] * data[:, i]

    dz = np.array(
        [surf_settings[n_dims] + random.uniform(-rand_range * 0.5, rand_range * 0.5) for _ in range(n_points)])

    data[:, n_dims] += dz

    return data


def distance_sum(x: np.ndarray, y: np.ndarray, k: float, b: float) -> float:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b при фиксированных k и b
    по формуле: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: значение параметра k (наклон)
    :param b: значение параметра b (смещение)
    :returns: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5
    """
    return np.sqrt(np.power((y - x * k + b), 2.0).sum())


def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
    значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
    F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: массив значений параметра k (наклоны)
    :param b: массив значений параметра b (смещения)
    :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    """
    return np.array([[distance_sum(x, y, k_i, b_i) for k_i in k.flat] for b_i in b.flat])


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Линейная регрессия.\n
    Основные формулы:\n
    yi - xi*k - b = ei\n
    yi - (xi*k + b) = ei\n
    (yi - (xi*k + b))^2 = yi^2 - 2*yi*(xi*k + b) + (xi*k + b)^2 = ei^2\n
    yi^2 - 2*(yi*xi*k + yi*b) + (xi^2 * k^2 + 2 * xi * k * b + b^2) = ei^2\n
    yi^2 - 2*yi*xi*k - 2*yi*b + xi^2 * k^2 + 2 * xi * k * b + b^2 = ei^2\n
    d ei^2 /dk = - 2*yi*xi + 2 * xi^2 * k + 2 * xi * b = 0\n
    d ei^2 /db = - 2*yi + 2 * xi * k + 2 * b = 0\n
    ====================================================================================================================\n
    d ei^2 /dk = (yi - xi * k - b) * xi = 0\n
    d ei^2 /db =  yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ(yi - xi * k) = n * b\n
    ====================================================================================================================\n
    Σyi - k * Σxi = n * b\n
    Σxi*yi - xi^2 * k - xi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*(Σyi - k * Σxi) / n = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*Σyi / n + k * (Σxi)^2 / n = 0\n
    Σxi*yi - Σxi*Σyi / n + k * ((Σxi)^2 / n - Σxi^2)  = 0\n
    Σxi*yi - Σxi*Σyi / n = -k * ((Σxi)^2 / n - Σxi^2)\n
    (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n) = k\n
    окончательно:\n
    k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
    b = (Σyi - k * Σxi) /n\n
    :param x: массив значений по x
    :param y: массив значений по y
    :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
    """
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_xx = (x * x).sum()
    n = x.size
    k = (sum_xy - sum_x * sum_y / n) / (sum_xx - sum_x * sum_x / n)
    return k, (sum_y - k * sum_x) / n


def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """
    Билинейная регрессия.\n
    Основные формулы:\n
    zi - (yi * ky + xi * kx + b) = ei\n
    zi^2 - 2*zi*(yi * ky + xi * kx + b) + (yi * ky + xi * kx + b)^2 = ei^2\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + ((yi*ky)^2 + 2 * (xi*kx*yi*ky + b*yi*ky) + (xi*kx + b)^2)\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx + b)^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b + b^2\n
    ====================================================================================================================\n
    d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
    d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
    d Σei^2 /db  = Σ-zi + yi*ky + xi*kx = 0\n
    ====================================================================================================================\n
    d Σei^2 /dkx / dkx = Σ xi^2\n
    d Σei^2 /dkx / dky = Σ xi*yi\n
    d Σei^2 /dkx / db  = Σ xi\n
    ====================================================================================================================\n
    d Σei^2 /dky / dkx = Σ xi*yi\n
    d Σei^2 /dky / dky = Σ yi^2\n
    d Σei^2 /dky / db  = Σ yi\n
    ====================================================================================================================\n
    d Σei^2 /db / dkx = Σ xi\n
    d Σei^2 /db / dky = Σ yi\n
    d Σei^2 /db / db  = n\n
    ====================================================================================================================\n
    Hesse matrix:\n
    || d Σei^2 /dkx / dkx;  d Σei^2 /dkx / dky;  d Σei^2 /dkx / db ||\n
    || d Σei^2 /dky / dkx;  d Σei^2 /dky / dky;  d Σei^2 /dky / db ||\n
    || d Σei^2 /db  / dkx;  d Σei^2 /db  / dky;  d Σei^2 /db  / db ||\n
    ====================================================================================================================\n
    Hesse matrix:\n
                   | Σ xi^2;  Σ xi*yi; Σ xi |\n
    H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                   | Σ xi;    Σ yi;    n    |\n
    ====================================================================================================================\n
                      | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
    grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                      | Σ-zi + yi*ky + xi*kx                |\n
    ====================================================================================================================\n
    Окончательно решение:\n
    |kx|   |1|\n
    |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
    | b|   |0|\n

    :param x: массив значений по x
    :param y: массив значений по y
    :param z: массив значений по z
    :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
    """
    sum_x = x.sum()
    sum_y = y.sum()
    sum_z = z.sum()
    sum_xy = (x * y).sum()
    sum_xx = (x * x).sum()
    sum_yy = (y * y).sum()
    sum_zy = (z * y).sum()
    sum_zx = (x * z).sum()

    hessian = np.array([[sum_xx, sum_xy, sum_x],
                        [sum_xy, sum_yy, sum_y],
                        [sum_x, sum_y, x.size]])

    hessian = np.linalg.inv(hessian)

    # grad = np.array([sum_xy + sum_xx - sum_zx, sum_yy + sum_xy - sum_zy, sum_y + sum_x - sum_z])

    # x_0 = np.array([1.0, 1.0, 0.0])

    # return x_0 - hessian @ grad
    return np.array([1.0, 1.0, 0.0]) - \
           hessian @ np.array([sum_xy + sum_xx - sum_zx, sum_yy + sum_xy - sum_zy, sum_y + sum_x - sum_z])


def n_linear_regression(data_rows: np.ndarray) -> np.ndarray:
    """
    H_ij = Σx_i * x_j, i in [0, rows - 1] , j in [0, rows - 1]
    H_ij = Σx_i, j = rows i in [rows, :]
    H_ij = Σx_j, j in [:, rows], i = rows

           | Σkx * xi^2    + Σky * xi * yi + b * Σxi - Σzi * xi|\n
    grad = | Σkx * xi * yi + Σky * yi^2    + b * Σyi - Σzi * yi|\n
           | Σyi * ky      + Σxi * kx                - Σzi     |\n

    x_0 = [1,...1, 0] =>

           | Σ xi^2    + Σ xi * yi - Σzi * xi|\n
    grad = | Σ xi * yi + Σ yi^2    - Σzi * yi|\n
           | Σxi       + Σ yi      - Σzi     |\n

    :param data_rows:  состоит из строк вида: [x_0,x_1,...,x_n, f(x_0,x_1,...,x_n)]
    :return:
    """
    s_rows, s_cols = data_rows.shape

    hessian = np.zeros((s_cols, s_cols,), dtype=float)

    grad = np.zeros((s_cols,), dtype=float)

    x_0 = np.zeros((s_cols,), dtype=float)

    s_cols -= 1

    # x_0[row] = 1.0: Устанавливается соответствующий элемент вектора x_0 равным 1,
    # чтобы учесть свободный член в модели линейной регрессии.

    for row in range(s_cols):
        x_0[row] = 1.0
        for col in range(row + 1):
            hessian[row, col] = np.dot(data_rows[:, row], data_rows[:, col])
            hessian[col, row] = np.dot(data_rows[:, row], data_rows[:, col])

    for i in range(s_cols + 1):
        hessian[i, s_cols] = (data_rows[:, i]).sum()
        hessian[s_cols, i] = (data_rows[:, i]).sum()

    hessian[s_cols, s_cols] = data_rows.shape[0]  # Заполняется последний элемент матрицы гессиана,
                                                  # представляющий общее количество строк данных.

    # Для каждой строки row, grad[row] вычисляется как сумма всех элементов в строке row матрицы гессиана (кроме последнего столбца)
    # минус скалярное произведение последнего столбца данных (значений целевой переменной) и строки row.
    for row in range(s_cols):
        grad[row] = hessian[row, 0: s_cols].sum() - np.dot(data_rows[:, s_cols], data_rows[:, row])

    # вычисляется как сумма всех элементов в последнем столбце матрицы гессиана (кроме последнего элемента)
    # минус сумма всех значений целевой переменной.
    grad[s_cols] = hessian[s_cols, 0: s_cols].sum() - data_rows[:, s_cols].sum()

    return x_0 - np.linalg.inv(hessian) @ grad


def poly_regression(x: np.ndarray, y: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Полином: y = Σ_j x^j * bj\n
    Отклонение: ei =  yi - Σ_j xi^j * bj\n
    Минимизируем: Σ_i(yi - Σ_j xi^j * bj)^2 -> min\n
    Σ_i(yi - Σ_j xi^j * bj)^2 = Σ_iyi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2\n
    условие минимума:\n d/dbj Σ_i ei = d/dbj (Σ_i yi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2) = 0\n
    :param x: массив значений по x
    :param y: массив значений по y
    :param order: порядок полинома
    :return: набор коэффициентов bi полинома y = Σx^i*bi
    """
    a_m = np.zeros((order, order,), dtype=float)
    c_m = np.zeros((order,), dtype=float)
    """

        for row in range(order):
            c_m[row] = (x**row * y).sum()
            for col in range(row + 1):
                a_m[row][col] = (x**(row + col)).sum()
                a_m[col][row] = a_m[row][col]
        return np.linalg.inv(a_m) @ c_m
    """
    # n = x.size
    _x_row = np.ones_like(x)
    for row in range(order):
        _x_row = _x_row if row == 0 else _x_row * x
        c_m[row] = np.dot(_x_row, y)  # (_x_row * y).sum()
        _x_col = np.ones_like(x)
        for col in range(row + 1):
            _x_col = _x_col if col == 0 else _x_col * x
            a_m[row][col] = np.dot(_x_col, _x_row) # .sum()
            a_m[col][row] = a_m[row][col]

    return np.linalg.inv(a_m) @ c_m


def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    :param x: массив значений по x\n
    :param b: массив коэффициентов полинома\n
    :returns: возвращает полином yi = Σxi^j*bj\n
    """
    result = b[0] + b[1] * x
    _x = x.copy()
    for i in range(2, b.size):
        _x *= x
        result += b[i] * _x
    return result


def quadratic_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:

    b = [x * x, x * y, y * y, x, y, np.array([1.0])]

    a_m = np.zeros((6, 6), dtype=float)

    b_c = np.zeros((6,), dtype=float)

    for row in range(6):
        b_c[row] = (b[row] * z).sum()
        for col in range(row + 1):
            a_m[row][col] = (b[row] * b[col]).sum()
            a_m[col][row] = a_m[row][col]

    a_m[5][5] = x.size # костыль, который я не придумал как убрать

    return np.linalg.inv(a_m) @ b_c


def distance_field_test():
    """
    Функция проверки поля расстояний:\n
    1) Посчитать тестовыe x и y используя функцию test_data\n
    2) Задать интересующие нас диапазоны k и b (np.linspace...)\n
    3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.\n
    4) Проанализировать результат (смысл этой картинки в чём...)\n
    :return:
    """
    x, y = test_data()
    k_, b_ = linear_regression(x, y)
    print(f"y(x) = {k_:1.5} * x + {b_:1.5}")
    k = np.linspace(-2.0, 2.0, 128, dtype=float)
    b = np.linspace(-2.0, 2.0, 128, dtype=float)
    z = distance_field(x, y, k, b)
    plt.imshow(z, extent=[k.min(), k.max(), b.min(), b.max()])
    plt.plot(k_, b_, 'r*')
    plt.xlabel("k")
    plt.ylabel("b")
    plt.grid(True)
    plt.show()


def linear_reg_test():
    """
    Функция проверки работы метода линейной регрессии:\n
    1) Посчитать тестовыe x и y используя функцию test_data\n
    2) Получить с помошью linear_regression значения k и b\n
    3) Вывести на графике x и y в виде массива точек и построить\n
       регрессионную прямую вида: y = k*x + b\n
    :return:
    """
    x, y = test_data()
    k, b = linear_regression(x, y)
    print(f"y(x) = {k:1.5} * x + {b:1.5}")
    plt.plot([0, 1.0], [b, k + b], 'g')
    plt.plot(x, y, 'r.')
    plt.show()


def bi_linear_reg_test():
    """
    Функция проверки работы метода билинейной регрессии:\n
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d\n
    2) Получить с помошью bi_linear_regression значения kx, ky и b\n
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить\n
       регрессионную плоскость вида:\n z = kx*x + ky*y + b\n
    :return:
    """
    # from matplotlib import cm
    x, y, z = test_data_2d()
    kx, ky, b = bi_linear_regression(x, y, z)
    print("\nbi linear regression test:")
    print(f"z(x, y) = {kx:1.5} * x + {ky:1.5} * y + {b:1.5}\n")
    x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    z_ = kx * x_ + y_ * ky + b
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(x, y, z, 'r.')
    surf = ax.plot_surface(x_, y_, z_, linewidth=0, antialiased=False, edgecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def poly_reg_test():
    """
    Функция проверки работы метода полиномиальной регрессии:\n
    1) Посчитать тестовыe x, y используя функцию test_data\n
    2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression\n
    3) Вывести на графике x и y в виде массива точек и построить\n
       регрессионную кривую. Для построения кривой использовать метод polynom\n
    :return:
    """
    print('\npoly regression test:')
    x, y = test_data()
    coefficients = poly_regression(x, y)
    y_ = polynom(x, coefficients)
    print(f"y(x) = {' + '.join(f'{coefficients[i]:.4} * x^{i}' for i in range(coefficients.size))}\n")
    plt.plot(x, y_, 'g')
    plt.plot(x, y, 'r.')
    plt.show()


def n_linear_reg_test():
    data = test_data_nd()
    print("\nn linear regression test:")
    coefficients = n_linear_regression(data)
    print(f"z(X) = {' + '.join(f'{coefficients[i]:.4} * x_{i}' if i != coefficients.size - 1 else f'{coefficients[i]:.4}' for i in range(coefficients.size))}\n")


def quadratic_reg_test():
    """
    """
    x, y, z = test_data_2d_poly()
    coeffs = quadratic_regression_2d(x, y, z)
    # y_ = polynom(x, coefficients)
    print('\n2d quadratic regression test:')
    print(f"z(x, y) = {coeffs[0]:1.3} * x^2 + {coeffs[1]:1.3} * x * y + {coeffs[2]:1.3} * y^2 + {coeffs[3]:1.3} * x + {coeffs[4]:1.3} * y + {coeffs[5]:1.3}")
    from matplotlib import cm
    x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    z_ = coeffs[0] * x_ * x_ + coeffs[1] * x_ * y_ + coeffs[2] * y_ * y_ + coeffs[3] * x_ + coeffs[4] * y_  + coeffs[5]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(x, y, z, 'r.')
    surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm, linewidth=0, antialiased=False, edgecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == "__main__":
    distance_field_test()
    linear_reg_test()
    bi_linear_reg_test()
    n_linear_reg_test()
    poly_reg_test()
    quadratic_reg_test()
