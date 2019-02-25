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

def nodes(xs, ys, n, x):
    s = [[xs[i], ys[i]] for i in range(len(xs))]
    s = np.array(sorted(s, key=lambda x: x[0]))
    xs, ys = s[:,0], s[:,1]
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
    
    return xs[start:stop+1], ys[start:stop+1]

def newton_polynomial(xs, ys, n, x):
    xs, ys = nodes(xs, ys, n, x)
    dd = divided_difference(xs, ys, n)
    xc, y = 1, ys[0]
    for i in range(n):
        xc *= x - xs[i]
        y += xc * dd[0][i]
    return y

def print_title(title):
    print('─' * 4, title, '─' * (80 - 4 - 1 - len(title) - 1))

z = 0
def assign_z_to(value):
    global z
    z = value

def main():
    print_title('List of tables:')
    list_tables()
    
    print_title('Menu:')
    filename = input('Input filename: ')
    title, xs, ys = load_table(filename)

    print_title('Table:')
    print_table(title, xs, ys)

    print_title('Menu:')
    x = float(input('Input x: '))
    n = int(input('Input n: '))

    y = newton_polynomial(xs, ys, n, x)
    if 'x' in title:
        eval('assign_z_to({})'.format(title))
    else:
        global z
        z = float(title)

    ae = fabs(z - y)
    re = ae / fabs(z)

    print('Interpolated value: {:.3f}'.format(y))
    print('Real value        : {:.3f}'.format(z))
    print('Absolute error    : {:.3f}'.format(ae))
    print('Relative error    : {:.3f}'.format(re))

if __name__ == '__main__':
    main()
