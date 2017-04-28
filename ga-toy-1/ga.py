# Program for ...
# Students: ...

import sys
import argparse
import random
import util
import copy
import math

DEBUG = True

class NString:
    @staticmethod
    def generate(n = 1):
        return NString(n)

    def __init__(self, args):
        self.n = args["n"]
        self.string = args["string"]
        if self.string is None:
            self.string = ""
            for i in xrange(self.n): 
                self.string += str(int(random.uniform(0, 1) + 0.5))

    def mutate(self, p):
        newString = ""
        for i in xrange(self.n):
            if util.flipCoin(p): newString += str(int(not int(self.string[i])))
            else:                newString += self.string[i]
        self.string = newString

    def cross(self, other):
        assert self.n == other.n, "Right now only handles crossover between equal length strings"
        if self.n >= 2:
            idx = int(math.floor(1 + random.uniform(0, 1) * (self.n - 2)))
            #print self.n, idx
            return [NString({"n" : self.n, "string" : self.string[:idx] + other.string[idx:]}), \
                    NString({"n" : self.n, "string" : other.string[:idx] + self.string[idx:]})]
        else:
            return [copy.copy(self), copy.copy(other)]
        

    def getStringLength(self):
        return self.n

    def countOnes(self):
        sum = 0
        for i in xrange(self.n):
            sum += int(self.string[i])
        return sum

    def countZeros(self):
        return self.n - self.countOnes()

    def __str__(self):
        return self.string

    def hash(self):
        return self.string

class GA:
    class Settings:
        def __init__(self):
            self.generationSize = 1
            self.percentKept    = 0.2
            self.probCross      = 0.7
            self.probMutate     = 0.1

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
                               for i in xrange(self.settings.populationSize)] 

    def advance(self):
        # build counter dictionary using fit for probability
        probDist = util.Counter()
        hashes = []
        for dna in self.population:
            if dna.hash() not in hashes:
                probDist[dna] += self.fit(dna)
                hashes.append(dna.hash())
                #print dna, self.fit(dna)
        
        newGen = probDist.sortedKeys()[:int(math.floor(self.settings.percentKept * self.settings.populationSize + random.uniform(0, 1)))]
        # pick floor(n + 1) dnas from parent generation

        for i in xrange((self.settings.populationSize + 1 - len(newGen)) // 2):
            parent1 = util.sampleFromCounter(probDist)
            parent2 = util.sampleFromCounter(probDist)
            next = []
            if util.flipCoin(self.settings.probCross):
                next = parent1.cross(parent2)
            else:
                next = [copy.copy(parent1), copy.copy(parent2)]
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
            string += "  " + str(dna) + "\n"
        return string

# Main wrapper 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Write me.')
    parser.add_argument('-n', action="store", dest="n", type=int, default=4)
    parser.add_argument('-pop', action="store", dest="pop", type=int, default=10)
    parser.add_argument('-p', action="store", dest="pop", type=int, default=10)
    parser.add_argument('-run', action="store", dest="run", type=int, default=5)
    parser.add_argument('-r', action="store", dest="run", type=int, default=5)
    parser.add_argument('-top', action="store", dest="top", type=int, default=1)
    parser.add_argument('-t', action="store", dest="top", type=int, default=1)
    parser.add_argument('-debug', action="store_true", default=False)

    args = parser.parse_args()
    DEBUG = args.debug

    dnaArgs = {"n": args.n, "string": None}
    gaArgs  = GA.Settings()
    gaArgs.populationSize = args.pop

    ga = GA(NString, dnaArgs, lambda l: 2**l.countOnes(), gaArgs)
    ga.initializeRandom()
    print ga

    for i in xrange(100):
        ga.advance()

    print ga.getTop()
