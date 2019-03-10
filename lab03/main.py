import numpy as np
from table import *

def divided_difference(xs, ys, n):
    def _dd(yi, yj, xi, xj):
        denom = 1e-12 if xi == xj else xi - xj
        return (yi - yj) / denom
    dd = np.zeros((n, n))
    for i in range(n):
        dd[i][0] = _dd(ys[i], ys[i + 1], xs[0], xs[1])
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = _dd(dd[i][j - 1], dd[i + 1][j - 1], xs[i], xs[i + j + 1])
    return dd

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

def indices(xs, n, x):
    xm = lower_bound(xs, x)
    rh = n // 2
    lh = n - rh

    start = xm - lh
    stop = xm + rh

    if start < 0:
        stop -= start
        start = 0
    elif stop >= len(xs):
        start -= stop - len(xs) + 1
        stop = len(xs) - 1

    return start, stop

def newton_polynomial(xs, ys, n, x):
    dd = divided_difference(xs, ys, n)
    xc, y = 1, ys[0]
    for i in range(n):
        xc *= x - xs[i]
        y += xc * dd[0][i]
    return y

def newton_polynomial2(xs, ys, zs, nx, ny, x, y):
    start_x, stop_x = indices(xs, nx, x)
    start_y, stop_y = indices(ys, ny, y)
    xs = xs[start_x:stop_x+1]
    ys = ys[start_y:stop_y+1]
    zs = zs[start_x:stop_x+1, start_y:stop_y+1]
    zx = [newton_polynomial(ys, zs[i], ny, y) for i in range(len(xs))]
    return newton_polynomial(xs, zx, nx, x)

def print_title(title):
    print('─' * 4, title, '─' * (80 - 4 - 1 - len(title) - 1))

g = 0
def assign_g_to(value):
    global g
    g = value

def main():
    print_title('List of tables:')
    list_tables()
    
    print_title('Menu:')
    filename = input('Input filename: ')
    title, xs, ys, zs = load_table(filename)

    print_title('Table:')
    print_table(title, xs, ys, zs)

    print_title('Menu:')
    x = float(input('Input x: '))
    y = float(input('Input y: '))
    nx = int(input('Input nx: '))
    ny = int(input('Input ny: '))

    z = newton_polynomial2(xs, ys, zs, nx, ny, x, y)
    eval('assign_g_to({})'.format(title))

    ae = fabs(g - z)
    re = ae / fabs(g)

    print('Interpolated value: {:.3f}'.format(z))
    print('Real value        : {:.3f}'.format(g))
    print('Absolute error    : {:.3f}'.format(ae))
    print('Relative error    : {:.3f}'.format(re))

if __name__ == '__main__':
    main()
