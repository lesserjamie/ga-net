import numpy
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def complexity(layer_sizes):
    count = 0

    for i in range(len(layer_sizes) - 1):
        count += layer_sizes[i] * layer_sizes[i + 1]

    return count

if __name__ == '__main__':
    #dataname = 'diabetes'
    #dataname = 'ionosphere'
    dataname = 'connect4'

    # without occam's razor
    filename = './output_' + dataname + '.txt'
    #data = numpy.genfromtxt(filename, delimiter=';', skip_header = 1,
    #                        names=('generation', 'fit', 'layers'), 
    #                        dtype=(float, float, list), 
    #                        converters = {2: lambda s: json.loads(str(s).strip("b'"))})

    # with occam's razor
    filenameO = './output_' + dataname + '_occams.txt'
    dataO = numpy.genfromtxt(filenameO, delimiter=';', skip_header = 1,
                             names=('generation', 'fit', 'layers'), 
                             dtype=(float, float, list), 
                             converters = {2: lambda s: json.loads(str(s).strip("b'"))})



    # AVERAGE FIT ###############################################################################

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.5)

    #ax.plot(data['generation'], data['fit'], 'g.', alpha=0.1)
    ax.plot(dataO['generation'], dataO['fit'], 'b.', alpha=0.1)
    #ax.plot(list(set(data['generation'])),
    #        [numpy.average([d['fit'] for d in data if d['generation'] == g]) for g in set(data['generation'])],
    #        'g', label = "Without Occam's")
    ax.plot(list(set(dataO['generation'])),
            [numpy.average([d['fit'] for d in dataO if d['generation'] == g]) for g in set(dataO['generation'])],
            'b', label = "With Occam's")

    ax.set_ylim(ymin=0, ymax=1)
    plt.xlabel('Generation')
    plt.ylabel('Cross Validation')
    plt.title(dataname[0].upper() + dataname[1:] + ' Data, Average Cross Validation')
    plt.legend(loc=4)

    fig.savefig('./' + dataname + '_avg_fit.png') 
    plt.close(fig)



    # BEST FIT ####################################################################################
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.5)

    #ax.plot(data['generation'], data['fit'], 'g.', alpha=0.1)
    ax.plot(dataO['generation'], dataO['fit'], 'b.', alpha=0.1)
    #ax.plot(list(set(data['generation'])),
    #        [numpy.max([d['fit'] for d in data if d['generation'] == g]) for g in set(data['generation'])],
    #        'g', label = "Without Occam's")
    ax.plot(list(set(dataO['generation'])),
            [numpy.max([d['fit'] for d in dataO if d['generation'] == g]) for g in set(dataO['generation'])],
            'b', label = "With Occam's")

    ax.set_ylim(ymin=0, ymax=1)
    plt.xlabel('Generation')
    plt.ylabel('Cross Validation')
    plt.title(dataname[0].upper() + dataname[1:] + ' Data, Max Cross Validation')
    plt.legend(loc=4)

    fig.savefig('./' + dataname + '_max_fit.png') 
    plt.close(fig)

    # AVG NUM NODES ####################################################################################
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.5)

    #ax.plot(data['generation'], [sum(l) for l in data['layers']], 'g.', alpha=0.1)
    ax.plot(dataO['generation'], [sum(l) for l in dataO['layers']], 'b.', alpha=0.1)
    #ax.plot(list(set(data['generation'])),
    #        [numpy.average([sum(d['layers']) for d in data if d['generation'] == g]) for g in set(data['generation'])],
    #        'g', label = "Without Occam's")
    ax.plot(list(set(dataO['generation'])),
            [numpy.average([sum(d['layers']) for d in dataO if d['generation'] == g]) for g in set(dataO['generation'])],
            'b', label = "With Occam's")
    
    ax.set_ylim(ymin=0)    
    plt.xlabel('Generation')
    plt.ylabel('Number of Nodes')
    plt.title(dataname[0].upper() + dataname[1:] + ' Data, Average Number of Nodes')
    plt.legend(loc=1)

    fig.savefig('./' + dataname + '_num_nodes.png') 
    plt.close(fig)

    # AVG COMPLEXITY #################################################################################
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.5)

    #ax.plot(data['generation'], [complexity(l) for l in data['layers']], 'g.', alpha=0.1)
    ax.plot(dataO['generation'], [complexity(l) for l in dataO['layers']], 'b.', alpha=0.1)
    #ax.plot(list(set(data['generation'])),
    #        [numpy.average([complexity(d['layers']) for d in data if d['generation'] == g]) for g in set(data['generation'])],
    #        'g', label = "Without Occam's")
    ax.plot(list(set(dataO['generation'])),
            [numpy.average([complexity(d['layers']) for d in dataO if d['generation'] == g]) for g in set(dataO['generation'])],
            'b', label = "With Occam's")
    
    ax.set_ylim(ymin=0)    
    plt.xlabel('Generation')
    plt.ylabel('Complexity')
    plt.title(dataname[0].upper() + dataname[1:] + ' Data, Complexity')
    plt.legend(loc=1)

    fig.savefig('./' + dataname + '_complexity.png') 
    plt.close(fig)
