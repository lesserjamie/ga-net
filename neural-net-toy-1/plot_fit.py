import numpy
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = numpy.genfromtxt('./output.txt', delimiter=';', skip_header = 1,
                            names=('generation', 'fit', 'layers'), 
                            dtype=(float, float, list), 
                            converters = {2: lambda s: \
                                              json.loads(str(s).strip("b'"))})

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylim([0, 1])
    ax.plot(data['generation'], data['fit'], 'bo')
    plt.xlabel('Generation')
    plt.ylabel('Fit')
    fig.savefig('./fit.png') 
    plt.close(fig)

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(data['generation'], [sum(l) for l in data['layers']], 'bo')
    plt.xlabel('Generation')
    plt.ylabel('Number of Nodes')
    fig.savefig('./layer_depth.png') 
    plt.close(fig)
