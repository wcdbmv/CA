import numpy as np
from math import *
import os

def table(start, stop, step, f):
    xs = np.arange(start, stop + step, step)
    ys = np.array(list(map(f, xs)))
    return xs, ys

def print_table(title, xs, ys):
    print('{:^31s}'.format(title))
    print('{:^15s}│{:^15s}'.format('x', 'y'))
    print('─' * 15 + '┼' + '─' * 15)
    for i in range(len(xs)):
        print('{:^15.3f}│{:^15.3f}'.format(xs[i], ys[i]))

def dump_table(filename, title, xs, ys):
    with open('tables/' + filename, 'w') as file:
        file.write(title + '\n')
        for i in range(len(xs)):
            file.write('{} {}\n'.format(xs[i], ys[i]))

def load_table(filename):
    with open('tables/' + filename, 'r') as file:
        title = file.readline().rstrip()
        xs, ys = [], []
        for line in file:
            x, y = map(float, line.split())
            xs.append(x)
            ys.append(y)
    return title, xs, ys

def list_tables():
    print('\n'.join(os.listdir('tables/')))

def main():
    f = input('Input f(x) = ')
    start = float(input('Input start: '))
    stop = float(input('Input stop: '))
    step = float(input('Input step: '))
    filename = input('Input filename: ')
    title = input('Input title: ')

    eval('dump_table(filename, title, *table(start, stop, step, lambda x: {}))'.format(f))

if __name__ == '__main__':
    main()
