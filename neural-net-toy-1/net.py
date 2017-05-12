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

DEBUG = True
OCCAMS = False

class Net:
    @staticmethod
    def generate(args):
        return Net(args)

    def __init__(self, args):
        self.trained = False
        self.data = args["data"]
        self.target = args["target"]
        self.pctValidate = 0.1
        self.fitValue   = None
        self.fitValueOccams = None

        if "fit_parameters" in args: 
            self.pctValidate = args["fit_parameters"]

        if "layer_sizes" in args:
            self.layer_sizes = args["layer_sizes"]
        elif "layer_parameters" in args:
            layer_depth = int(random.uniform(1, args["layer_parameters"][0]))
            layer_sizes = [int(random.uniform(2, args["layer_parameters"][1])) for i in range(layer_depth)]
            self.layer_sizes = layer_sizes
        else:
            print("Going with defaults.")
            layer_depth = int(random.uniform(1, 4))
            layer_sizes = [int(random.uniform(2, 14)) for i in range(layer_depth)]
            self.layer_sizes = layer_sizes

        self.net = None

    def mutate(self, p):
        new_layer_sizes = []
        for size in self.layer_sizes:
            if random.uniform(0,1) < p:
                self.trained = False
                offset = 2*int(2*random.uniform(0, 1)) - 1
                #print("offset " + str(offset))
                new_layer_sizes.append(max(size + offset, 1))
            else:
                new_layer_sizes.append(size)
        self.layer_sizes = new_layer_sizes

    def cross(self, other):
        idx1 = int(random.uniform(0, len(self.layer_sizes)))
        idx2 = int(random.uniform(0, len(other.layer_sizes)))

        return [Net({"data" : self.data, "target" : self.target, "fit_parameters": self.pctValidate,
                     "layer_sizes" : self.layer_sizes[:idx1] + other.layer_sizes[idx2:]}),
                Net({"data" : self.data, "target" : self.target, "fit_parameters": self.pctValidate,
                     "layer_sizes" : other.layer_sizes[:idx2] + self.layer_sizes[idx1:]})]

    def train(self):
        if not self.trained:
            self.net = MLPClassifier(tuple(self.layer_sizes),solver='lbfgs')
            self.net.fit(self.data, self.target)
            self.trained = True
        
    def fit(self, occams):
        if self.fitValueOccams is not None and occams:
            return self.fitValueOccams
        if self.fitValue is not None and not occams:
            return self.fitValue

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

    def __str__(self):
        return str(self.layer_sizes)

    def hash(self):
        return json.dumps(self.layer_sizes)


class GA:
    class Settings:
        def __init__(self):
            self.generationSize = 1
            self.percentKept    = 0.2
            self.probCross      = 0.7
            self.probMutate     = 0.5

    def __init__(self, dnaClass, dnaArgs, fit, settings = None):
        self.settings = GA.Settings()
        if settings is not None: self.settings = settings
        self.population = []
        self.dnaClass   = dnaClass
        self.dnaArgs    = dnaArgs
        self.fit        = fit
        self.generation = 0
        self.csv = "generation,fit,hash\n"

    def initializeRandom(self):
        self.population = [self.dnaClass.generate(self.dnaArgs)\
                               for i in range(self.settings.populationSize)] 

    def advance(self):
        # build counter dictionary using fit for probability
        # now don't need to worry about hashes because no duplicates in a generation
        probDist = util.Counter()
        for dna in self.population:
            probDist[dna] += self.fit(dna, OCCAMS)
        
        newGen = self.population[:int(math.floor(self.settings.percentKept * self.settings.populationSize + random.uniform(0, 1)))]
        hashes = [dna.hash() for dna in newGen]

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

        if len(newGen) > self.settings.populationSize: newGen = newGen[:-1]
        assert len(newGen) == self.settings.populationSize           

        self.population = newGen
        self.population.sort(key=lambda a: self.fit(a, OCCAMS), reverse=True)
        self.generation += 1

        for dna in self.population:
            self.csv += (str(self.generation) + ";" 
                         + str(self.fit(dna, False)) + ";"
                         + str(dna.hash()) + "\n")

    def getCSV(self):
        return self.csv

    def getTop(self):
        return self.population[0]
           
    def __str__(self):
        string = "Generation: " + str(self.generation) + "\n"
        for dna in self.population:
            string += "  " + str(dna) + ", fit is: " + str(self.fit(dna, OCCAMS)) + "\n"
        best = self.getTop()
        string += "Best model is " + str(best) + " with fit " + str(self.fit(best, OCCAMS)) + "\n"
        return string

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

# Main wrapper 
if __name__ == '__main__':

    """parser = argparse.ArgumentParser(description='Write me.')
    parser.add_argument('-data', action="store", dest="data", default="data/yellow-small.data")

    args = parser.parse_args()
    filename = args.data"""

    # Parses Pima Indians diabetes data
    f = open("data/pima-indians-diabetes.data")
    lines = list(f)
    diabetesData = []
    diabetesTarget = []
    for line in lines:
        line.replace('\n','')
        values = line.split(',')
        dataPoint = [float(values[i]) for i in range(len(values) - 1)]
        diabetesData.append(dataPoint)
        diabetesTarget.append(values[len(values) - 1])
    f.close()
    data = numpy.array(diabetesData)
    target = numpy.array(diabetesTarget)

    # Parses ionosphere data
    f = open("data/ionosphere.data")
    lines = list(f)
    ionosphereData = []
    ionosphereTarget = []
    for line in lines:
        values = line.split(',')
        dataPoint = [float(values[i]) for i in range(len(values) - 1)]
        ionosphereData.append(dataPoint)
        if 'g\n' in values:
            dataPointTarget = 1
        else:
            dataPointTarget = 0
        ionosphereTarget.append(dataPointTarget)
    #data = numpy.array(ionosphereData)
    #target = numpy.array(ionosphereTarget)
    f.close()

    # Parses Connect 4 data
    f = open("data/connect-4.data")
    lines = list(f)
    connect4Data = []
    connect4Target = []
    for line in lines:
        values = line.split(',')
        dataPoint = [0 if values[i] == 'b' else (1 if values[i] == 'x' else -1) for i in range(len(values) - 1)]
        connect4Data.append(dataPoint)
        if 'win\n' in values:
            dataPointTarget = 1
        elif 'loss\n' in values:
            dataPointTarget = -1
        else:
            dataPointTarget = 0
        connect4Target.append(dataPointTarget)
    #data = numpy.array(connect4Data)
    #target = numpy.array(connect4Target)
    f.close()


    digits = datasets.load_digits()
    #data = digits.data
    #target = digits.target
    #data = balloonsData
    #target = balloonsTarget
     
    #print("________________________________________________________")

    dnaArgs = {"data": data, 
	       "target": target, 
               "layer_parameters": (5, 15),
               "fit_parameters": 0.1}

    gaArgs  = GA.Settings()
    gaArgs.populationSize = 20

    ga = GA(Net, dnaArgs, lambda l, o: l.fit(o), gaArgs)
    ga.initializeRandom()
    #print(ga)
    
    for i in range(50):
        print(i)
        ga.advance()

        #print(ga)

    #print("________________________________________________________")

    print(ga.getCSV())
    output = open("output.txt", "w")
    output.write(ga.getCSV())
    output.close()
