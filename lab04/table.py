import os

def print_table(xs, ys, ws):
    print('{:^15s}│{:^15s}│{:^15s}'.format('x', 'y', 'w'))
    print('─' * 15 + '┼' + '─' * 15 + '┼' + '─' * 15)
    for i in range(len(xs)):
        print('{:^15.3f}│{:^15.3f}│{:^15.3f}'.format(xs[i], ys[i], ws[i]))

def load_table(filename):
    with open('tables/' + filename, 'r') as file:
        xs, ys, ws = [], [], []
        for line in file:
            x, y, w = map(float, line.split())
            xs.append(x)
            ys.append(y)
            ws.append(w)
    return xs, ys, ws

def list_tables():
    print('\n'.join(os.listdir('tables/')))
