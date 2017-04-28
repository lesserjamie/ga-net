import random

# Adjustable parameters
NUM_NEURAL_NETS = 0
NUM_TIMES_TRAINED_PER_ITERATION = 0
NUM_NETS_SAVED = 0
NUM_NETS_MUTATED = 0
NUM_NETS_BRED = 0
NUM_ITERATIONS_GENETIC_ALGORITHM = 0

# Abstract specifications of methods required for genetic algorithm
class NeuralNet:
    # Initializes a neural net based on the possible attributes and values
    def __init__(self, attributes):
        pass

    # Returns a mutated version of this neural net
    def mutate(self):
        pass

    # Combines this neural net with other, returning the result
    def combine(self, other):
        pass

    # Trains this neural net on the provided data for numTimes iterations
    def train(self, data, numTimes = 1):
        pass

    # Calculates how good this neural net is, based on the data
    # Higher fitness is better
    # Will call into other neural net functions, like how many errors it makes
    def fitness(self, data):
        pass

# Given a set of training data, a set of test data, and the attributes of the data points with possible values, determines the best neural net
# The test data is used when comparing the neural nets to each other in the genetic algorithm, and is such is a part of learning the neural net
# For this reason, there should be actual test data kept separate and used at the end to see how our learned neural net performs
def geneticAlgorithm(trainingData, testData, attributes):
    # Initialize a pool of neural nets
    neuralNets = [NeuralNet(attributes) for i in range(NUM_NEURAL_NETS)]
    # Run a number of iterations
    for i in range(NUM_ITERATIONS_GENETIC_ALGORITHM):
        # Train each neural net some number of times on the provided data
        for neuralNet in neuralNets:
            neuralNet.train(trainingData, NUM_TIMES_TRAINED_PER_ITERATION)
        # Calculate a score for each neural net
        neuralNetWeights = [neuralNet.fitness(testingData) for neuralNet in neuralNets]
        # Save some number of them, chosen at random, with a higher chance of choosing better ones
        savedNeuralNets = [weightedChoice(neuralNets, neuralNetWeights) for i in range(NUM_NETS_SAVED)]
        # Mutate some number of them, chosen at random, with a higher chance of choosing better ones
        mutatedNeuralNets = [weightedChoice(neuralNets, neuralNetWeights).mutate() for i in range(NUM_NETS_MUTATED)]
        # Combine randomly chosen pairs, with a higher chance of choosing better ones, to create some number of new ones
        # Currently could select the same neural net twice, resulting in saving an additional copy of that neural net
        bredNeuralNets = [weightedChoice(neuralNets, neuralNetWeights).combine(weightedChoice(neuralNets, neuralNetWeights)) for i in range(NUM_NETS_BRED)]
        # Update our pool of neural nets to be those either saved, mutated, or bred
        neuralNets = []
        neuralNets.extend(savedNeuralNets)
        neuralNets.extend(mutatedNeuralNets)
        neuralNets.extend(bredNeuralNets)
    # After the appropriate number of iterations, choose the best
    bestNet = None
    bestScore = float("inf")
    for neuralNet in neuralNets:
        fitness = neuralNet.fitness(testingData)
        if fitness > bestScore:
            bestScore = fitness
            bestNet = neuralNet
    return bestNet

# Takes a list of choices and list of corresponding weights and returns a choice chosen at random
def weightedChoice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    for i in range(len(choices)):
        r = r - weights[i]
        if r <= 0:
            return choices[i]
    # Should never happen
    return choices[0]
