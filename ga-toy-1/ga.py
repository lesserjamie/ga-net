# Program for ...
# Students: ...

import sys
import argparse
import random

DEBUG = True

class NString:
    @staticmethod
    def generate(n = 1):
        return NString(n)

    def __init__(self, n = 1):
        self.n = n
        self.string = ""
        for i in xrange(self.n): 
            self.string += str(int(random.uniform(0, 1) + 0.5))

    def mutate(self, num):
        assert num <= self.n, "Cannot have more mutations than there are places in the string"
        mutate = [1] * num + [0] * (self.n - num)
        random.shuffle(mutate)
        newString = ""
        for i in xrange(self.n):
            if mutate[i]: newString += str(int(not int(self.string[i])))
            else:         newString += self.string[i]
        self.string = newString


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
    def __init__(self, dna, fit, populationSize = 1, runs = 1, top = 1, probMutation = 0.5):
        self.populationSize = populationSize
        self.runs = runs
        self.top = top
        self.population = [dna.generate() for i in xrange(self.populationSize)]
        self.fit = fit

    #def advance(self):
        

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
    test = NString(4)
    print test
    test.mutate(2)
    print test

    ga = GA(NString, lambda l: l.countOnes(), args.pop, args.run, args.top)
