
from mpl_toolkits import mplot3d

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from sys import exit
from os.path import expanduser

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def main_f():
    cmap = get_cmap(15)
    l = hlp.load__p("{}/car_model/debug/1000000_p.csv".format(expanduser("~")))

    print(l)
    matrix = []
    for item in l:
        size = len(item["traj"])
        path = np.zeros([size, 3])
        for idx, pos in enumerate(item["traj"]):
            path[idx] = pos[0]
        matrix.append(path)

    # exit(0)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    color = ['red', 'green', 'gold', 'black', 'lime', 'violet', 'plum',
             'orange', 'navy', 'salmon']
    for i, p in enumerate(matrix):
        zline = np.array(p[:, 2])
        xline = np.array(p[:, 1])
        yline = np.array(p[:, 0])
        ax.plot3D(xline, yline, zline, color[i % len(color)])

    ax.view_init(azim=0, elev=100)

    plt.savefig("{}/car_model/tmp.png".format(expanduser("~")))
    plt.show()

if __name__ == "__main__":
   main_f()