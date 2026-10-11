from layer import Layer

class Network:
    def __init__(self, shape): #creates all the layers with their appropriate number of neurons
        self.shape = shape
        self.layers: list[Layer] = []
        for i in range(1, len(self.shape)):
            if i == len(self.shape) - 1: newLayer = Layer(self.shape[i-1], self.shape[i], None)
            else: newLayer = Layer(self.shape[i-1], self.shape[i], "relu")
            self.layers.append(newLayer)
    
    def forward(self, inputs): #forwards all the layers using the previous layer as input
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, incoming_gradients):
        gradients = incoming_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)