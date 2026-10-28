from enum import Enum
from math import exp, tanh
from random import random, seed

seed(0)

def rng():
    return 2*random() - 1

class Network:
    def __init__(self, layers: list[int], activations : list[ActivationFunctions], softmax : bool=False) -> Network:
        self.network = []
        self.softmax = softmax
        if len(layers) != len(activations):
            raise Exception
        else:
            for activate, size in enumerate(layers):
                layer = []
                for i in range(size):
                    node = Node(activations[activate])
                    if self.network != []:
                        parents = self.network[-1]
                        node.setParent(parents)
                        for parent in parents:
                            parent.addChild(node, node.parents[parent])
                    
                    layer.append(node)
                self.network.append(layer)

    def runInput(self, vector : list[float]) -> list[float]:
        newVector = []
        for i, var in enumerate(vector):
            newVector.append(self.network[0][i].compute(var))

        for layer in self.network[1:]:
            print(newVector)
            nextVector = []
            for node in layer:
                nextVector.append(node.compute(newVector))

            newVector = nextVector
        if self.softmax:
            return softmax(nextVector)
        else:
            return nextVector
        
class ActivationFunctions(Enum):
    RELU = 0
    LEAKYRELU = 1
    SIGMOID = 2
    TANH = 3

    def selectActivation(value):
        if value == ActivationFunctions.RELU:
            return relu
        elif value == ActivationFunctions.LEAKYRELU:
            return leakyrelu
        elif value == ActivationFunctions.SIGMOID:
            return sigmoid
        elif value == ActivationFunctions.TANH:
            return tanh
        
def sigmoid(x):
    return 1/(1+exp(-x))

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def leakyrelu(x, alpha=0.1):
    if x > 0:
        return x
    else:
        return alpha * x

def softmax(v):
    sum = 0
    for i in v:
        sum += exp(i)

    ans = []
    for j in v:
        ans.append(exp(j)/sum)

    return ans

class Node:
    def __init__(self, activation):
        self.parents = {}
        self.childs = {}
        self.bias = rng()
        self.activationFunction = activation

    def addChild(self, child : Node, weight : float):
        self.childs[child] = weight

    def addParent(self, parent : Node):
        self.parents[parent] = rng()

    def setParent(self, parents : list[Node]):
        for x in parents:
            self.parents[x] = rng()

    def compute(self, vector) -> float:
        if self.parents == {}:
            sum = vector
        else:
            if len(self.parents) != len(vector):
                print("Dimension Mismatch")
                raise Exception

            i = 0
            sum = 0
            for i, (key, value) in enumerate(self.parents.items()):
                sum += value * vector[i]

        return ActivationFunctions.selectActivation(self.activationFunction)(sum + self.bias)

newNetwork = Network([10, 5, 3], [ActivationFunctions.RELU, ActivationFunctions.RELU, ActivationFunctions.SIGMOID], softmax=True)

for i in newNetwork.network:
    for j in i:
        print(f"Child: {j.childs}")
        print(f"Parent: {j.parents}")
        print(f"Bias: {j.bias}")
        print()

    print("\n")

print(newNetwork.runInput([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))