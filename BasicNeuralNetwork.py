import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class BasicNeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 8)
        self.weights2   = np.random.rand(8, 8)
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, iterations):
        for i in range(iterations):
            self.feedforward()
            self.backprop()
            #progress = i * 100 / iterations
            print("Iteration %d / %d" % (i, iterations), end = "\r")
        print("Iteration %d / %d" % (i, iterations), end = "\r")
        print()

    def checkvalue(self, input):
        self.input = input
        self.feedforward()
        left = []
        color = []
        max = -1
        maxindex = -1
        for i in range(len(self.output)):
            if self.output[i]>max:
                max = self.output[i]
                maxindex = i
            left.append(i)
            color.append('red')
        color[maxindex] = 'green'
        height = self.output
        tick_label = left
        plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = color) 
        plt.xlabel('x - axis') 
        plt.ylabel('y - axis') 
        plt.title('Test result: ' + str(maxindex)) 
        plt.show()
