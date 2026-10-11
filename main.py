from vocabulary import Vocabulary
from network import Network
from layer import softmax
from loss import cross_entropy
import numpy as np

text = "the cat eats the cat sleeps"
vocab = Vocabulary(text)

size = len(vocab.vocab)
network = Network([size, 8, size])

logits = network.forward(vocab.one_hot("the")) # starts the network with the word the
probabilities = softmax(logits)

target_id = vocab.vocab["cat"]  #returns id of the word
loss = cross_entropy(probabilities, target_id) #punishes extremely wrong ones
print(f"Before: {loss}")
correct_word_vector = vocab.one_hot("cat") #[0,1,0,0,0]
hidden_outputs = network.layers[0].outputs

output_logits_gradients = probabilities - correct_word_vector

output_weight_gradients = np.outer(output_logits_gradients, hidden_outputs) #weight gradient = logit gradient x input
output_bias_gradients = output_logits_gradients * 1 #the 1 is redundant but good for theory

hidden_output_gradients = network.layers[1].weights.T @ output_logits_gradients # for each node = [total of {the next layers gradients x the weights connecting them to the node}]
hidden_weighted_sum_gradients = (network.layers[0].weighted_sums > 0) * hidden_output_gradients #find the weight and bias gradients for the hidden layer now

hidden_weight_gradients = np.outer(hidden_weighted_sum_gradients, network.layers[0].inputs)
hidden_bias_gradients = hidden_weighted_sum_gradients * 1

learning_rate = 0.01

network.layers[1].weights -= learning_rate * output_weight_gradients
network.layers[1].biases -= learning_rate * output_bias_gradients

network.layers[0].weights -= learning_rate * hidden_weight_gradients
network.layers[0].biases -= learning_rate * hidden_bias_gradients

logits = network.forward(vocab.one_hot("the"))
probabilities = softmax(logits)
target_id = vocab.vocab["cat"]  #returns id of the word
loss = cross_entropy(probabilities, target_id) #punishes extremely wrong ones
print(f"After: {loss}")
