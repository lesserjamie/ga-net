# Program for ...
# Students: ...

import sys
import argparse
import random
import copy
import math
import numpy
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

DEBUG = True

class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        #compare = lambda x, y:  sign(y[1] - x[1])
        #sortedItems.sort(cmp=compare)
        sorted(sortedItems,key=lambda x: -x[1])
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

def normalize(vectorOrCounter):
    """
    normalize a vector or counter by dividing each value by the sum of all values
    """
    normalizedCounter = Counter()
    if type(vectorOrCounter) == type(normalizedCounter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0: return counter
        for key in counter.keys():
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        s = float(sum(vector))
        if s == 0: return vector
        return [el / s for el in vector]

def sample(distribution, values = None):
    if type(distribution) == Counter:
        items = sorted(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    choice = random.random()
    i, total= 0, distribution[0]
    while choice > total:
        i += 1
        if i >= len(distribution):
            print(i)
            print(distribution)
            print(total)
            print(choice)
        total += distribution[i]
    return values[i]

def sampleFromCounter(ctr):
    items = sorted(ctr.items(), key = lambda x: -x[1])
    return sample([v for k,v in items], [k for k,v in items])


class Net:
    @staticmethod
    def generate(args):
        return Net(args)

    def __init__(self, args):
        self.trained = False
        self.train_data =args["train_data"]
        self.test_data = args["test_data"]
        self.train_target = args["train_target"]
        self.test_target = args["test_target"]
        
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
                offset = 2*int(2*random.uniform(0, 1)) - 1
                #print("offset " + str(offset))
                new_layer_sizes.append(max(size + offset, 1))
            else:
                new_layer_sizes.append(size)
        self.layer_sizes = new_layer_sizes

    def cross(self, other):
        idx1 = int(random.uniform(0, len(self.layer_sizes)))
        idx2 = int(random.uniform(0, len(other.layer_sizes)))

        return [Net({"train_data" : self.train_data, "test_data" : self.test_data, "train_target" : self.train_target, "test_target" : self.test_target, "layer_sizes" : self.layer_sizes[:idx1] + other.layer_sizes[idx2:]}),
                Net({"train_data" : self.train_data, "test_data" : self.test_data, "train_target" : self.train_target, "test_target" : self.test_target, "layer_sizes" : other.layer_sizes[:idx2] + self.layer_sizes[idx1:]})]
        #return [copy.deepcopy(self), copy.deepcopy(other)]

    def train(self):
        if not self.trained:
            self.net = MLPClassifier(tuple(self.layer_sizes),max_iter=1000,solver='lbfgs')
            self.net.fit(self.train_data, self.train_target)
            self.trained = True
        
    def test(self):
        assert self.net is not None
        return self.net.score(self.test_data, self.test_target)

    def fit(self):
        self.train()
        return self.test()
	
    def __str__(self):
        return str(self.layer_sizes)

    def hash(self):
        return self.layer_sizes


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

    def initializeRandom(self):
        self.population = [self.dnaClass.generate(self.dnaArgs)\
                               for i in range(self.settings.populationSize)] 

    def advance(self):
        # build counter dictionary using fit for probability
        probDist = Counter()
        hashes = []
        for dna in self.population:
            if dna.hash() not in hashes:
                probDist[dna] += self.fit(dna)
                hashes.append(dna.hash())
                #print dna, self.fit(dna)
        
        newGen = probDist.sortedKeys()[:int(math.floor(self.settings.percentKept * self.settings.populationSize + random.uniform(0, 1)))]
        # pick floor(n + 1) dnas from parent generation

        for i in range((self.settings.populationSize + 1 - len(newGen)) // 2):
            parent1 = sampleFromCounter(probDist)
            parent2 = sampleFromCounter(probDist)
            next = []
            if random.uniform(0,1) < self.settings.probCross:
                next = parent1.cross(parent2)
            else:
                next = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            for dna in next:
                dna.mutate(self.settings.probMutate)
            newGen.extend(next)

        if len(newGen) > self.settings.populationSize: newGen = newGen[:-1]
        assert len(newGen) == self.settings.populationSize           

        self.population = newGen
        self.population.sort(key=lambda a: self.fit(a), reverse=True)
        self.generation += 1

    def getTop(self):
        return self.population[0]

           
    def __str__(self):
        string = "Generation: " + str(self.generation) + "\n"
        for dna in self.population:
            string += "  " + str(dna) + ", fit is: " + str(self.fit(dna)) + "\n"
        best = self.getTop()
        string += "Best model is " + str(best) + " with fit " + str(self.fit(best)) + "\n"
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
        


    digits = datasets.load_digits()
    #data = digits.data
    #target = digits.target
    #data = balloonsData
    #target = balloonsTarget
    
    pctTest = 0.1
    split = [int(len(data)*pctTest), len(data) - int(len(data)*pctTest)]
    split_target = numpy.split(target, split)
    test_target = split_target[0]
    train_target = split_target[1]
    split_data = numpy.split(data, split)
    test = split_data[0]
    train = split_data[1]
    print("________________________________________________________")

    dnaArgs = {"test_target": test_target, "train_target": train_target, 
	       "test_data": test, "train_data": train, "layer_parameters": (4, 14)}
    net = Net(dnaArgs)
    net.train()
    print(net.test())

    gaArgs  = GA.Settings()
    gaArgs.populationSize = 12

    ga = GA(Net, dnaArgs, lambda l: l.fit(), gaArgs)
    ga.initializeRandom()
    print(ga)
    
    for i in range(6):
        ga.advance()
        print(ga)

