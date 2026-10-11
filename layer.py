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
