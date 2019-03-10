import numpy as np
from math import *
import os

# Note:
#
# table in file:
# <title>
# xs[0] ... xs[-1]
# ys[0] ... ys[-1]
# zs[      0][0] ... zs[      0][len(ys)]
#      ...       ...        ...
# zs[len(xs)][0] ... zs[len(xs)][len(ys)]
#
#
# and print_table:
#         <title>
#          │  xs[0]       ...    xs[-1]
# ─────────┼───────────────────────────────
#   ys[0]  │  zs[0][0]    ...   zs[-1][0]
#    ...   │    ...       ...     ...
#  ys[-1]  │  zs[0][-1]   ...   zs[-1][-1]

def table(x_start, x_stop, x_step, y_start, y_stop, y_step, f):
    xs = np.arange(x_start, x_stop + x_step, x_step)
    ys = np.arange(y_start, y_stop + y_step, y_step)
    zs = np.zeros((len(xs), len(ys)))
    for i in range(len(xs)):
        for j in range(len(ys)):
            zs[i][j] = f(xs[i], ys[j])
    return xs, ys, zs

def print_table(title, xs, ys, zs):
    table_head = '         │' + '{:^9.3f} ' * (len(xs))
    table_data = '{:^9.3f}│' + '{:^9.3f} ' * len(xs)
    print('\t{}'.format(title))
    print(table_head.format(*xs))
    print('─' * 9 + '┼' + '─' * len(xs) * 10)
    for i in range(len(ys)):
        print(table_data.format(ys[i], *zs[:,i]))

def dump_table(filename, title, xs, ys, zs):
    with open('tables/' + filename, 'w') as file:
        file.write(title + '\n')
        for vs in (xs, ys):
            for v in vs:
                file.write('{:9.3f} '.format(v))
            file.write('\n')
        for i in range(len(xs)):
            for j in range(len(ys)):
                file.write('{:9.3f} '.format(zs[i][j]))
            file.write('\n')

def load_table(filename):
    with open('tables/' + filename, 'r') as file:
        title = file.readline().rstrip()
        xs = np.array(list(map(float, file.readline().split())))
        ys = np.array(list(map(float, file.readline().split())))
        zs = np.array([list(map(float, line.split())) for line in file])
    return title, xs, ys, zs

def list_tables():
    print('\n'.join(os.listdir('tables/')))

def main():
    f = input('Input f(x, y) = ')
    x_start = float(input('Input x_start: '))
    x_stop = float(input('Input x_stop: '))
    x_step = float(input('Input x_step: '))
    y_start = float(input('Input y_start: '))
    y_stop = float(input('Input y_stop: '))
    y_step = float(input('Input y_step: '))
    filename = input('Input filename: ')
    title = input('Input title: ')

    eval('dump_table(filename, title, *table(x_start, x_stop, x_step, y_start, y_stop, y_step, lambda x, y: {}))'.format(f))

if __name__ == '__main__':
    main()
