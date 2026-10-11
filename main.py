from vocabulary import Vocabulary
from network import Network
from layer import softmax
from loss import cross_entropy
import numpy as np

text = "the cat eats the cat sleeps"
vocab = Vocabulary(text)

size = len(vocab.vocab)
network = Network([size, 8, size])

logits = network.forward(vocab.one_hot("the"))
probabilities = softmax(logits)

print(f"Before: {cross_entropy(probabilities, vocab.vocab["cat"])}")
gradients = probabilities - vocab.one_hot("cat")

network.backward(gradients)
network.update(0.01)

logits = network.forward(vocab.one_hot("the"))
probabilities = softmax(logits)
print(f"After: {cross_entropy(probabilities, vocab.vocab["cat"])}")