# Program for ...
# Students: ...

import util
import json
import sys
import argparse
import random
import copy
import math
import numpy
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

# Controls whether or not an Occam's Razor principle is used to keep the neural nets at lower complexity
OCCAMS = False

# Neural net wrapper
class Net:
    @staticmethod
    def generate(args):
        return Net(args)

    # args provides the data points with relevant attribute values, the target values as which these data points should be classified and optionally information about the structure of the hidden layers
    def __init__(self, args):
        self.data = args["data"]
        self.target = args["target"]
        # Fraction of the data left out for testing
        self.pctValidate = 0.1
        self.fitValue   = None
        self.fitValueOccams = None

        if "fit_parameters" in args: 
            self.pctValidate = args["fit_parameters"]

        # Determine the structure of the neural net
        # If exact layer sizes are specified in args, use those
        # Otherwise, if a max number of hidden layers and max number of hidden nodes per layer are specified, choose randomly up to those maxes
        # Otherwise, go with the defaults of up to four hidden layers and 14 hidden nodes in each layer
        if "layer_sizes" in args:
            self.layer_sizes = args["layer_sizes"]
        elif "layer_parameters" in args:
            layer_depth = int(random.uniform(1, args["layer_parameters"][0]))
            layer_sizes = [int(random.uniform(2, args["layer_parameters"][1])) for i in range(layer_depth)]
            self.layer_sizes = layer_sizes
        else:
            layer_depth = int(random.uniform(1, 4))
            layer_sizes = [int(random.uniform(2, 14)) for i in range(layer_depth)]
            self.layer_sizes = layer_sizes

    # Mutates the structure of the neural net
    # With probability p for each hidden layer
    def mutate(self, p):
        new_layer_sizes = []
        for size in self.layer_sizes:
            if random.uniform(0,1) < p:
                offset = 2*int(2*random.uniform(0, 1)) - 1
                new_layer_sizes.append(max(size + offset, 1))
            else:
                new_layer_sizes.append(size)
        self.layer_sizes = new_layer_sizes

    # Crosses over two nets to produce two new structures
    # A random split point is chosen in each net to partition the layers into two sets
    # One net then takes the first half of one and the second half of the other, while the other net takes the other sets
    def cross(self, other):
        idx1 = int(random.uniform(0, len(self.layer_sizes)))
        idx2 = int(random.uniform(0, len(other.layer_sizes)))

        return [Net({"data" : self.data, "target" : self.target, "fit_parameters": self.pctValidate,
                     "layer_sizes" : self.layer_sizes[:idx1] + other.layer_sizes[idx2:]}),
                Net({"data" : self.data, "target" : self.target, "fit_parameters": self.pctValidate,
                     "layer_sizes" : other.layer_sizes[:idx2] + self.layer_sizes[idx1:]})]

    # Provides the fit of this net
    # If occams is false, then this is simply the cross-validation success rate
    # If occams is true, then this is modified to reward a net for having a simpler structure
    def fit(self, occams):
        # First, if we've already calculated the fit, that value is stored and we can simply return it
        if self.fitValueOccams is not None and occams:
            return self.fitValueOccams
        if self.fitValue is not None and not occams:
            return self.fitValue

        # Cross validation part; we partition the data into subsets, whose size is set according to pctValidate
        # We then proceed through these subsets, leaving one at a time out, training a net on the remaining data, and then testing on the left out set
        # The final fit is the average success rate across all the tests run
        numTests = min(int(math.floor(1.0 / self.pctValidate)), len(data))
        split = []
        error = 0.0

        for i in range(numTests):
            error += len(self.data) * self.pctValidate
            split.append(int(math.floor(error)))
            error -= math.floor(error)
        
        assert numpy.sum(split) == len(self.data)
        assert len(split) == numTests
        split = [numpy.sum(split[:i+1]) for i in range(len(split) - 1)]

        target_partition = numpy.split(self.target, split)
        data_partition = numpy.split(self.data, split)
        scoreSum = 0.0

        for i in range(numTests):
            train_data = numpy.concatenate(tuple(data_partition[:i] + data_partition[i + 1:]))
            train_target = numpy.concatenate(tuple(target_partition[:i] + target_partition[i + 1:]))
            net = MLPClassifier(tuple(self.layer_sizes),solver='lbfgs')
            net.fit(train_data, train_target)            
            scoreSum += net.score(data_partition[i], target_partition[i])

        self.fitValue = scoreSum / float(numTests)
        # Occam's razor implementation; fitValue modified based on how many nodes used, with fewer being better, with the 56 coming from the defaults chosen above
        self.fitValueOccams = self.fitValue + .3 * (56.0 - sum(self.layer_sizes)) / 56.0
        if occams:
            return self.fitValueOccams
        else:
            return self.fitValue

    # Returns the structure of the net
    def __str__(self):
        return str(self.layer_sizes)

    # Again, structure of the net
    def hash(self):
        return json.dumps(self.layer_sizes)

# The genetic algorithm
class GA:
    # Settings defines how many neural nets are kept from one generation to the next, what the probability of crossover is, and what the probability of mutation is
    class Settings:
        def __init__(self):
            self.generationSize = 1
            self.percentKept    = 0.2
            self.probCross      = 0.7
            self.probMutate     = 0.5

    # Initializes the genetic algorithm with a class (Net, in this case), arguments which are fed to that class, a fit function, and optionally settings to override the defaults
    def __init__(self, dnaClass, dnaArgs, fit, settings = None):
        self.settings = GA.Settings()
        if settings is not None: self.settings = settings
        self.population = []
        self.dnaClass   = dnaClass
        self.dnaArgs    = dnaArgs
        self.fit        = fit
        self.generation = 0
        self.csv = "generation,fit,hash\n"

    # Generates populationSize initial instances of the class, each of which is initialized with the provided arguments
    def initializeRandom(self):
        self.population = [self.dnaClass.generate(self.dnaArgs)\
                               for i in range(self.settings.populationSize)] 

    # Advances the genetic algorithm one step
    # Done by calculating the fit of each net, and then generating a new population
    def advance(self):
        # Each net will be weighted based on its fit
        # build counter dictionary using fit for probability
        # now don't need to worry about hashes because no duplicates in a generation
        probDist = util.Counter()
        for dna in self.population:
            probDist[dna] += self.fit(dna, OCCAMS)
        
        # Keep some of the best, according to the percentKept in settings
        # hashes is to ensure we do not end up with two copies of the same structure
        newGen = self.population[:int(math.floor(self.settings.percentKept * self.settings.populationSize + random.uniform(0, 1)))]
        hashes = [dna.hash() for dna in newGen]

        # Until we have a full generation, keep adding new nets
        # Done by selecting two nets at random, possibly crossing them over, and then possibly mutating each of the two structures
        # A net is only added if it is not a duplicate structure
        while len(newGen) < self.settings.populationSize:
            parent1 = util.sampleFromCounter(probDist)
            parent2 = util.sampleFromCounter(probDist)
            next = []
            if random.uniform(0,1) < self.settings.probCross:
                next = parent1.cross(parent2)
            else:
                next = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            for dna in next:
                dna.mutate(self.settings.probMutate)
                if dna.hash() not in hashes:
                    newGen.append(dna)
                    hashes.append(dna.hash())

        # Ensure we have the right size population
        if len(newGen) > self.settings.populationSize: newGen = newGen[:-1]
        assert len(newGen) == self.settings.populationSize           

        self.population = newGen
        self.population.sort(key=lambda a: self.fit(a, OCCAMS), reverse=True)
        self.generation += 1

        # For each net, add the relevant information to a csv for later analysis
        # Even if we are using Occam's, we still want this data to be the actual succes rate at classifying the data
        for dna in self.population:
            self.csv += (str(self.generation) + ";" 
                         + str(self.fit(dna, False)) + ";"
                         + str(dna.hash()) + "\n")

    # Fetches the csv
    def getCSV(self):
        return self.csv

    # Gets the current best neural net
    def getTop(self):
        return self.population[0]
           
    def __str__(self):
        string = "Generation: " + str(self.generation) + "\n"
        for dna in self.population:
            string += "  " + str(dna) + ", fit is: " + str(self.fit(dna, False)) + "\n"
        best = self.getTop()
        string += "Best model is " + str(best) + " with fit " + str(self.fit(best, False)) + "\n"
        return string


def loadData(name):
    if name == "balloons":
        # Balloons data
        balloonsData = numpy.array([[1,0,1,0,1,0,1,0],
                                    [1,0,1,0,1,0,1,0],
                                    [1,0,1,0,1,0,0,1],
                                    [1,0,1,0,0,1,1,0],
                                    [1,0,1,0,0,1,0,1],
                                    [1,0,0,1,1,0,1,0],
                                    [1,0,0,1,1,0,1,0],
                                    [1,0,0,1,1,0,0,1],
                                    [1,0,0,1,0,1,1,0],
                                    [1,0,0,1,0,1,0,1],
                                    [0,1,1,0,1,0,1,0],
                                    [0,1,1,0,1,0,1,0],
                                    [0,1,1,0,1,0,0,1],
                                    [0,1,1,0,0,1,1,0],
                                    [0,1,1,0,0,1,0,1],
                                    [0,1,0,1,1,0,1,0],
                                    [0,1,0,1,1,0,1,0],
                                    [0,1,0,1,1,0,0,1],
                                    [0,1,0,1,0,1,1,0],
                                    [0,1,0,1,0,1,0,1]])
        balloonsTarget = numpy.array([1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0])
        return (balloonsData, balloonsTarget)

    if name == "digits":
        digits = datasets.load_digits()
        return (digits.data, digits.target)

    if name == "connect-4":
        # Parses Connect 4 data
        f = open("data/connect-4.data")
        lines = list(f)
        data = []
        target = []
        for line in lines:
            values = line.split(',')
            dataPoint = [0 if values[i] == 'b' else (1 if values[i] == 'x' else -1) for i in range(len(values) - 1)]
            data.append(dataPoint)
            if 'win\n' in values:
                dataPointTarget = 1
            elif 'loss\n' in values:
                dataPointTarget = -1
            else:
                dataPointTarget = 0
            target.append(dataPointTarget)

        f.close()
        return (numpy.array(data), numpy.array(target))

    if name == "diabetes": 
        f = open("data/pima-indians-diabetes.data")
        lines = list(f)
        data = []
        target = []
        for line in lines:
            line.replace('\n','')
            values = line.split(',')
            dataPoint = [float(values[i]) for i in range(len(values) - 1)]
            data.append(dataPoint)
            target.append(values[len(values) - 1])
        f.close()
        return (numpy.array(data), numpy.array(target))

    else:
        f = open("data/ionosphere.data")
        lines = list(f)
        data = []
        target = []
        for line in lines:
            values = line.split(',')
            dataPoint = [float(values[i]) for i in range(len(values) - 1)]
            data.append(dataPoint)
            if 'g\n' in values:
                dataPointTarget = 1
            else:
                dataPointTarget = 0
            target.append(dataPointTarget)
        f.close()
        return (numpy.array(data), numpy.array(target))


# Main wrapper 
if __name__ == '__main__':
    data, target = loadData("connect-4")

    dnaArgs = {"data": data, 
	       "target": target, 
               "layer_parameters": (5, 15),
               "fit_parameters": 0.1}

    gaArgs  = GA.Settings()
    gaArgs.populationSize = 20

    ga = GA(Net, dnaArgs, lambda l, o: l.fit(o), gaArgs)
    ga.initializeRandom()
    
    for i in range(50):
        print(i)
        ga.advance()

    output = open("output.txt", "w")
    output.write(ga.getCSV())
    output.close()
