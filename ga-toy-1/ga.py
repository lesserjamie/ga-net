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
            print self.n, idx
            return [NString({"n" : self.n, "string" : self.string[:idx] + other.string[idx + 1:]}), \
                    NString({"n" : self.n, "string" : other.string[:idx] + self.string[idx + 1:]})]
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

class GA:
    def __init__(self, dna, dnaArgs, fit, populationSize = 1, runs = 1, top = 1, probMutation = 0.5):
        self.populationSize = populationSize
        self.runs = runs
        self.top = top
        self.population = [dna.generate(dnaArgs) for i in xrange(self.populationSize)]
        self.fit = fit
        self.probCross = 0.5

    def advance(self):
        # build counter dictionary using fit for probability
        probDist = util.Counter()
        for dna in self.population:
            probDist[dna] += 1
        

        newGen = []
        # pick floor(n + 1) dnas from parent generation
        for i in xrange((self.populationSize + 1) // 2):
            parent1 = util.sampleFromCounter(probDist)
            parent2 = util.sampleFromCounter(probDist)
            if util.flipCoin(self.probCross):
                newGen.extend(parent1.cross(parent2))

        for dna in newGen:
           print dna
           
    #def __str__(self):
                    

# Main wrapper 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Write me.')
    parser.add_argument('-n', action="store", dest="n", type=int, default=8)
    parser.add_argument('-pop', action="store", dest="pop", type=int, default=10)
    parser.add_argument('-p', action="store", dest="pop", type=int, default=10)
    parser.add_argument('-run', action="store", dest="run", type=int, default=5)
    parser.add_argument('-r', action="store", dest="run", type=int, default=5)
    parser.add_argument('-top', action="store", dest="top", type=int, default=1)
    parser.add_argument('-t', action="store", dest="top", type=int, default=1)
    parser.add_argument('-debug', action="store_true", default=False)

    args = parser.parse_args()
    print args.n

    DEBUG = args.debug

    dnaArgs = {"n": args.n, "string": None}

    ga = GA(NString, dnaArgs, lambda l: l.countOnes(), args.pop, args.run, args.top)
    ga.advance()
