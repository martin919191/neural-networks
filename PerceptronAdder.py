# This file contains the two classes that represents a perceptron and a neural network class that represents 
# an adder circuit.

class Perceptron:
    # init function receives the initial weights and bias for that perceptron
    def __init__(self, inputWeight1, inputWeight2, bias):
        self.inputWeight1 = inputWeight1
        self.inputWeight2 = inputWeight2
        self.bias = bias
    
    # output function receives the two input values for that perceptron and calculates the output (0 or 1), 
    # based on the weights and the bias
    def output(self, input1, input2):
        return 1 if ((input1 * -self.inputWeight1) + (input2 * -self.inputWeight2) + self.bias ) > 0 else 0

class neuralNetworkAdder:
    # init function creates all perceptrons needed in the neural network with specific weights and biases.
    # by default, all weights are 2 and biases are 3. This is because, with those values, the perceptron will 
    # behave as if it was a NAND gate, main gate for an adder circuit.
    def __init__(self):
        self.perceptronSumLayerA1 = Perceptron(2,2,3)
        self.perceptronSumLayerB1 = Perceptron(2,2,3)
        self.perceptronSumLayerB2 = Perceptron(2,2,3)
        self.perceptronCarLayerA1 = Perceptron(2,2,3)
        self.perceptronSumLayerO = Perceptron(2,2,3)
        self.input = [0,0]
        self.output = [0,0]

    
    def sum(self):
        outputLayerA = [self.perceptronSumLayerA1.output(self.input[0], self.input[1])]
        outputLayerB = [
            self.perceptronSumLayerB1.output(self.input[0], outputLayerA[0]),
            self.perceptronSumLayerB2.output(self.input[1], outputLayerA[0])
        ]
        outputLayerSum = self.perceptronSumLayerO.output(outputLayerB[0], outputLayerB[1])
        outputLayerCar = self.perceptronCarLayerA1.output(outputLayerA[0], outputLayerA[0])

        self.output = [outputLayerCar, outputLayerSum]
        



