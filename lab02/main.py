import numpy as np
from math import *
from table import *

def lower_bound(lst, key):
    left = -1
    right = len(lst)
    while left + 1 < right:
        middle = (left + right) // 2
        if lst[middle] >= key:
            right = middle
        else:
            left = middle
    return right

def spline_interpolation(xs, ys, x):
    j = lower_bound(xs, x)

    n = len(xs)
    h = np.zeros((n))
    A = np.zeros((n))
    B = np.zeros((n))
    D = np.zeros((n))
    F = np.zeros((n))
    a = np.zeros((n))
    b = np.zeros((n))
    c = np.zeros((n + 1))
    d = np.zeros((n))
    xi = np.zeros((n + 1))
    eta = np.zeros((n + 1))

    for i in range(1, n):
        h[i] = xs[i] - xs[i-1]

    for i in range(2, n):
        A[i] = h[i-1]
        B[i] = -2 * (h[i-1] + h[i])
        D[i] = h[i]
        F[i] = -3 * ((ys[i] - ys[i-1]) / h[i] - (ys[i-1] - ys[i-2]) / h[i-1])

    for i in range(2, n):
        xi[i+1] = D[i] / (B[i] - A[i] * xi[i])
        eta[i+1] = (A[i] * eta[i] + F[i]) / (B[i] - A[i] * xi[i])

    for i in range(n - 2, -1, -1):
        c[i] = xi[i] * c[i+1] + eta[i+1]

    for i in range(1, n):
        a[i] = ys[i-1]
        b[i] = (ys[i] - ys[i-1]) / h[i] - h[i] / 3 * (c[i+1] + 2 * c[i])
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    return a[j] \
         + b[j] * (x - xs[j-1]) \
         + c[j] * ((x - xs[j-1]) ** 2) \
         + d[j] * ((x - xs[j-1]) ** 3)

def print_title(title):
    print('─' * 4, title, '─' * (80 - 4 - 1 - len(title) - 1))

z = 0
def assign_z_to(value):
    global z
    z = value

def main():
    print_title('List of tables')
    list_tables()

    print_title('Menu:')
    filename = input('Input filename: ')
    title, xs, ys = load_table(filename)

    print_title('Table:')
    print_table(title, xs, ys)

    print_title('Menu:')
    x = float(input('Input x: '))

    y = spline_interpolation(xs, ys, x)
    eval('assign_z_to({})'.format(title))

    ae = fabs(z - y)
    re = ae / fabs(z)

    print('Interpolated value: {:.3f}'.format(y))
    print('Real value        : {:.3f}'.format(z))
    print('Absolute error    : {:.3f}'.format(ae))
    print('Relative error    : {:.3f}'.format(re))

if __name__ == '__main__':
    main()
