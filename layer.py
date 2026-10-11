import numpy as np

def relu(inputs):
    if isinstance(inputs, (list, np.ndarray)):
        outputs = []
        for num in inputs:
            if num < 0:
                num = 0
            outputs.append(num)
        return np.array(outputs)
    else:
        if inputs < 0:
            inputs = 0
        return inputs

def softmax(inputs): #expects numpy array
    inputs = inputs - np.max(inputs)
    exponentials = np.exp(inputs)
    total = np.sum(exponentials)
    return exponentials / total

class Layer:
    def __init__(self, n_inputs, n_neurons, activation="relu"):
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sums = (inputs @ self.weights.T) + self.biases
        if self.activation == "relu":
            self.outputs = relu(self.weighted_sums)
        else:
            self.outputs = self.weighted_sums
        return self.outputs

    def backward(self, incoming_gradients): #receives the gradients from the next layer 
        if self.activation == "relu": self.weighted_sum_gradients = (self.weighted_sums > 0) * incoming_gradients #if relu then multiply by relu derivatives 
        else: self.weighted_sum_gradients = incoming_gradients #else nothing changes

        self.weight_gradients = np.outer(self.weighted_sum_gradients, self.inputs) #multiply by the inputs to get the weight gradients
        self.bias_gradients = self.weighted_sum_gradients * 1 #dont multiply by anything because the neuron calculation only multiplies the weight and input, not the bias

        self.input_gradients = self.weights.T @ self.weighted_sum_gradients #multiply by the weights to get the input gradients, just like you multiply by the input to get the weight gradients
        #outer is used for weight gradients because each weight carries its own gradient
        #@ {matrix multiplication} is used for input gradients because each input affects multiple things
        #it also {adds the things it effects together to get the overall gradient for that input} per node
        return self.input_gradients #returns the input gradients, then sent into the previous layer as incoming gradients

    def update(self, learning_rate):
        self.weights = self.weights - (learning_rate * self.weight_gradients)
        self.biases = self.biases - (learning_rate * self.bias_gradients)
