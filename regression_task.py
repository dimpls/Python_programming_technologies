import matplotlib.pyplot as plt
from typing import Tuple, Union
import numpy as np
import random

class Regression:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Regression class is static class")

    @staticmethod
    def rand_in_range(rand_range: Union[float, Tuple[float, float]] = 1.0) -> float:
        if isinstance(rand_range, float):
            return random.uniform(-0.5 * rand_range, 0.5 * rand_range)
        if isinstance(rand_range, tuple):
            return random.uniform(rand_range[0], rand_range[1])
        return random.uniform(-0.5, 0.5)

    @staticmethod
    def test_data_along_line(k: float = 1.0, b: float = 0.1, arg_range: float = 1.0,
                             rand_range: float = 0.05, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует линию вида y = k * x + b + dy, где dy - аддитивный шум с амплитудой half_disp
        :param k: наклон линии
        :param b: смещение по y
        :param arg_range: диапазон аргумента от 0 до arg_range
        :param rand_range: диапазон шума данных
        :param n_points: количество точек
        :return: кортеж значений по x и y
        """
        x_step = arg_range / (n_points - 1)
        return np.array([i * x_step for i in range(n_points)]),\
               np.array([i * x_step * k + b + Regression.rand_in_range(rand_range) for i in range(n_points)])

    @staticmethod
    def second_order_surface_2d(surf_params:
                                Tuple[float, float, float, float, float, float] = (1.0, -2.0, 3.0, 1.0, 2.0, -3.0),
                                args_range: float = 1.0, rand_range: float = .1, n_points: int = 1000) -> \
                                Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует набор тестовых данных около поверхности второго порядка.
        Уравнение поверхности:
        z(x,y) = a * x^2 + x * y * b + c * y^2 + d * x + e * y + f
        :param surf_params: 
        :param surf_params [a, b, c, d, e, f]:
        :param args_range x in [x0, x1], y in [y0, y1]:
        :param rand_range:
        :param n_points:
        :return:
        """
        x = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        y = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        dz = np.array([surf_params[5] + Regression.rand_in_range(rand_range) for _ in range(n_points)])
        return x, y, surf_params[0] * x * x + surf_params[1] * y * x + surf_params[2] * y * y + \
               surf_params[3] * x + surf_params[4] * y + dz

    @staticmethod
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
        x = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        y = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
        dz = np.array([b + Regression.rand_in_range(rand_range) for _ in range(n_points)])
        return x, y, x * kx + y * ky + dz

    @staticmethod
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
        n_dims = surf_settings.size - 1
        data = np.zeros((n_points, n_dims + 1,), dtype=float)
        for i in range(n_dims):
            data[:, i] = np.array([Regression.rand_in_range(args_range) for _ in range(n_points)])
            data[:, n_dims] += surf_settings[i] * data[:, i]
        dz = np.array([surf_settings[n_dims] + Regression.rand_in_range(rand_range) for _ in range(n_points)])
        data[:, n_dims] += dz
        return data

    @staticmethod
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

    @staticmethod
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
        return np.array([[Regression.distance_sum(x, y, k_i, b_i) for k_i in k.flat] for b_i in b.flat])

    @staticmethod
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

        sum_xi = np.sum(x)
        sum_yi = np.sum(y)
        sum_xi_squared = np.sum(x ** 2)
        sum_xi_yi = np.sum(x * y)

        n = len(x)

        k = (sum_xi_yi - sum_xi * sum_yi / n) / (sum_xi_squared - sum_xi * sum_xi / n)
        b = (sum_yi - k * sum_xi) / n

        return k, b

    @staticmethod
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
        n = len(x)

        sum_xi = np.sum(x)
        sum_yi = np.sum(y)
        sum_zi = np.sum(z)
        sum_xi_squared = np.sum(x * x)
        sum_xi_yi = np.sum(x * y)
        sum_yi_squared = np.sum(y * y)
        sum_xi_zi = np.sum(x * z)
        sum_yi_zi = np.sum(y * z)

        hessian = np.array([[sum_xi_squared, sum_xi_yi, sum_xi],
                            [sum_xi_yi, sum_yi_squared, sum_yi],
                            [sum_xi, sum_yi, n]])

        grad = np.array([sum_xi_yi + sum_xi_squared - sum_xi_zi, sum_yi_squared + sum_xi_yi - sum_yi_zi,
                         sum_yi + sum_xi - sum_zi])
        """
        np.array([1.0, 1.0, 0.0]) представляет начальное приближение для параметров kx, ky и b. 
        Это может быть любое начальное приближение, и здесь оно выбрано как [1.0, 1.0, 0.0].
        
        np.linalg.inv(hessian) представляет обратную матрицу Гессе.
        Обратная матрица Гессе используется для определения шага, который минимизирует функцию ошибки.
        
        grad представляет градиент функции ошибки. Градиент указывает направление наибыстрейшего увеличения функции,
        и поскольку мы хотим минимизировать функцию ошибки, мы двигаемся в направлении, противоположном градиенту.
        """
        return np.array([1.0, 1.0, 0.0]) - np.linalg.inv(hessian) @ grad

    @staticmethod
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
                # скалярного произведения соответствующих столбцов данных (из data_rows)
                # для каждой комбинации столбцов (row, col) и (col, row)
                hessian[row, col] = hessian[col, row] = np.dot(data_rows[:, row], data_rows[:, col])

        for i in range(s_cols + 1):
            # вычисляет сумму значений в каждом столбце данных
            # и устанавливает их в соответствующие элементы матрицы Гессиана
            hessian[i, s_cols] = hessian[s_cols, i] = (data_rows[:, i]).sum()

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

    @staticmethod
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
        a_m = np.zeros((order, order,), dtype=float)  # будет использоваться для хранения коэффициентов матрицы,
                                                    # связанных с полиномиальным уравнением
        c_m = np.zeros((order,), dtype=float) # для хранения коэффициентов вектора y.

        _x_row = np.ones_like(x) # Этот массив будет использоваться для хранения степеней x в текущей строке полинома.

        for row in range(order):
            _x_row = _x_row if row == 0 else _x_row * x
            c_m[row] = np.dot(_x_row, y)
            _x_col = np.ones_like(x)
            for col in range(row + 1):
                _x_col = _x_col if col == 0 else _x_col * x
                a_m[row][col] = a_m[col][row] = np.dot(_x_col, _x_row)

        return np.linalg.inv(a_m) @ c_m

    @staticmethod
    def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        :param x: массив значений по x\n
        :param b: массив коэффициентов полинома\n
        :returns: возвращает полином yi = Σxi^j*bj\n
        """
        result = b[0] + b[1] * x
        for i in range(2, b.size):
            result += b[i] * x ** i
        return result

    @staticmethod
    def quadratic_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Генерирует набор коэффициентов поверхности второго порядка. Уравнение поверхности:
        z(x,y) = a * x^2 + x * y * b + c * y^2 + d * x + e * y + f
        Поверхность максимальна близка ко всем точкам их набора.
        Получить коэффициенты можно по формуле:
        C = A^-1 * B
        C = {a, b, c, d, e, f}^T (вектор столбец искомых коэффициентов)
        Далее введём обозначения:
        x_i - i-ый элемент массива x
        y_i - i-ый элемент массива y
        z_i - i-ый элемент массива z
        C = {a, b, c, d, e, f}^T (вектор столбец искомых коэффициентов)
        B = {Σ xi^2 * zi,
             Σ xi * yi * zi,
             Σ yi^2 * zi,
             Σ xi * zi,
             Σ yi * zi,
             Σ zi} - (вектор свободных членов)
        Чтобы построить матрицу A введём новую матрицу D, составленную из условий:
        D = { x^2 | x * y | y^2 | x | y | 1 }.
        Строка этой матрицы имеет вид:
        di = { xi^2, xi * yi, yi^2, xi, yi, 1 }.

        Матричный элемент матрицы A выражается из матрицы D следующим образом:
        a_ij = (D[:,i], D[:,j]), где (*, *) - скалярное произведение.
        Матрица A - симметричная и имеет размерность 6x6.
        :param x:
        :param y:
        :param z:
        :return:
        """
        n = len(x)  # Количество наблюдений

        # Создаем матрицу D
        D = np.column_stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)])

        # Рассчитываем матрицу A
        A = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                A[i, j] = np.sum(D[:, i] * D[:, j])

        # Рассчитываем вектор B
        B = np.array([
            np.sum(x ** 2 * z),
            np.sum(x * y * z),
            np.sum(y ** 2 * z),
            np.sum(x * z),
            np.sum(y * z),
            np.sum(z)
        ])

        # Решаем систему уравнений для получения коэффициентов
        coefficients = np.linalg.solve(A, B)

        return coefficients

    @staticmethod
    def distance_field_example():
        """
        Функция проверки поля расстояний:\n
        1) Посчитать тестовыe x и y используя функцию test_data\n
        2) Задать интересующие нас диапазоны k и b (np.linspace...)\n
        3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.\n
        4) Проанализировать результат (смысл этой картинки в чём...)\n
        :return:
        """
        print("distance field test:")
        x, y = Regression.test_data_along_line()
        k_, b_ = Regression.linear_regression(x, y)
        print(f"y(x) = {k_:1.5} * x + {b_:1.5}\n")
        k = np.linspace(-2.0, 2.0, 128, dtype=float)
        b = np.linspace(-2.0, 2.0, 128, dtype=float)
        z = Regression.distance_field(x, y, k, b)
        plt.imshow(z, extent=[k.min(), k.max(), b.min(), b.max()])
        plt.plot(k_, b_, 'r*')
        plt.xlabel("k")
        plt.ylabel("b")
        plt.grid(True)
        plt.show()

    @staticmethod
    def linear_reg_example():
        """
        Функция проверки работы метода линейной регрессии:\n
        1) Посчитать тестовыe x и y используя функцию test_data\n
        2) Получить с помошью linear_regression значения k и b\n
        3) Вывести на графике x и y в виде массива точек и построить\n
           регрессионную прямую вида: y = k*x + b\n
        :return:
        """
        print("linear reg test:")
        x, y = Regression.test_data_along_line()

        k, b = Regression.linear_regression(x, y)

        print(f"y(x) = {k:1.5} * x + {b:1.5}")

        # Строим график
        plt.scatter(x, y, label='Точки данных')
        plt.plot(x, k * x + b, color='red', label='Регрессионная прямая')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Линейная регрессия')
        plt.legend()
        plt.show()

    @staticmethod
    def bi_linear_reg_example():
        """
        Функция проверки работы метода билинейной регрессии:\n
        1) Посчитать тестовыe x, y и z используя функцию test_data_2d\n
        2) Получить с помошью bi_linear_regression значения kx, ky и b\n
        3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить\n
           регрессионную плоскость вида:\n z = kx*x + ky*y + b\n
        :return:
        """
        x, y, z = Regression.test_data_2d()

        kx, ky, b = Regression.bi_linear_regression(x, y, z)

        print("\nbi linear regression test:")
        print(f"z(x, y) = {kx:1.5} * x + {ky:1.5} * y + {b:1.5}\n")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, label='Точки данных')

        # Генерируем сетку для построения регрессионной плоскости
        x_range = np.linspace(min(x), max(x), 100)
        y_range = np.linspace(min(y), max(y), 100)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = kx * x_mesh + ky * y_mesh + b

        ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, label='Регрессионная плоскость')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Билинейная регрессия')
        plt.show()

    @staticmethod
    def poly_reg_example():
        """
        Функция проверки работы метода полиномиальной регрессии:\n
        1) Посчитать тестовыe x, y используя функцию test_data\n
        2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression\n
        3) Вывести на графике x и y в виде массива точек и построить\n
           регрессионную кривую. Для построения кривой использовать метод polynom\n
        :return:
        """
        print('\npoly regression test:')
        x, y = Regression.test_data_along_line()

        coefficients = Regression.poly_regression(x, y)

        y_ = Regression.polynom(x, coefficients)

        print(f"y(x) = {' + '.join(f'{coefficients[i]:.4} * x^{i}' for i in range(coefficients.size))}")
        plt.plot(x, y_, 'g')
        plt.plot(x, y, 'r.')
        plt.show()

    @staticmethod
    def n_linear_reg_example():
        print("\nn linear regression test:")

        # Генерируем тестовые данные для n-мерной линейной регрессии
        data_rows = Regression.test_data_nd()

        # Получаем коэффициенты n-мерной линейной регрессии
        coefficients = Regression.n_linear_regression(data_rows)

        # Выводим результаты
        print("Коэффициенты n-мерной линейной регрессии:")
        for i, coeff in enumerate(coefficients):
            print(f"b{i} = {coeff}")

        # Строим график для n-мерной линейной регрессии
        # В данном случае, поскольку у нас n-мерная линейная регрессия,
        # мы можем построить график для двух произвольных признаков.
        plt.scatter(data_rows[:, 0], data_rows[:, 1], label='Точки данных (n-мерная линейная регрессия)')
        plt.xlabel('Признак 1')
        plt.ylabel('Признак 2')
        plt.title('n-мерная линейная регрессия')
        plt.legend()
        plt.show()

    @staticmethod
    def quadratic_reg_example():
        """
        """
        x, y, z = Regression.second_order_surface_2d()
        coefficients = Regression.quadratic_regression_2d(x, y, z)
        print('\n2d quadratic regression test:')
        print(
            f"z(x, y) = {coefficients[0]:1.3} * x^2 + {coefficients[1]:1.3} * x * y + {coefficients[2]:1.3} * y^2 + {coefficients[3]:1.3} * x + {coefficients[4]:1.3} * y + {coefficients[5]:1.3}")
        from matplotlib import cm
        x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
        z_ = coefficients[0] * x_ * x_ + coefficients[1] * x_ * y_ + coefficients[2] * y_ * y_ + coefficients[3] * x_ + coefficients[4] * y_ + coefficients[
            5]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot(x, y, z, 'r.')
        surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm, linewidth=0, antialiased=False, edgecolor='none',
                               alpha=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == "__main__":
    Regression.distance_field_example()
    Regression.linear_reg_example()
    Regression.bi_linear_reg_example()
    Regression.n_linear_reg_example()
    Regression.poly_reg_example()
    Regression.quadratic_reg_example()

