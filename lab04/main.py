from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from table import *

def phi(x, coefs):
    return sum(coefs[i] * x**i for i in range(len(coefs)))

def draw(coefs, xs, ys):
    cx = np.arange(xs[0], xs[-1] + 1e-10, 0.1)
    plt.plot([x for x in cx], [phi(x, coefs) for x in cx])
    for i in range(len(xs)):
        plt.scatter(x=xs[i], y=ys[i], alpha=0.3)
    plt.show()

def gauss_seidel(A, B):
    A, X, n = A.copy(), B.copy(), len(A)
    for i in range(n):
        for j in range(i + 1, n):
            sep = A[j][i] / A[i][i]
            for k in range(n):
                A[j][k] -= A[i][k] * sep
            X[j] -= X[i] * sep
    for i in range(n - 1, -1, -1):
        for k in range(i + 1, n):
            X[i] -= A[i][k] * X[k]
        X[i] /= A[i][i]
    return X

def dot_product(xs, px, ys, py, ws):
    return sum(pow(xs[i], px) * pow(ys[i], py) * ws[i] for i in range(len(xs)))

def matrix(xs, ys, ws, n):
    A = np.array([[dot_product(xs, i, xs, j, ws) for j in range(n + 1)] for i in range(n + 1)])
    B = np.array([[dot_product(ys, 1, xs, i, ws)] for i in range(n + 1)])
    return A, B

def print_title(title):
    print('─' * 4, title, '─' * (80 - 4 - 1 - len(title) - 1))

def main():
    print_title('List of tables:')
    list_tables()

    print_title('Menu:')
    filename = input('Input filename: ')
    xs, ys, ws = load_table(filename)

    print_title('Table:')
    print_table(xs, ys, ws)

    print_title('Menu:')
    n = int(input('Input n: '))

    A, B = matrix(xs, ys, ws, n)
    coefs = gauss_seidel(A, B)

    draw(coefs, xs, ys)

if __name__ == '__main__':
    main()
