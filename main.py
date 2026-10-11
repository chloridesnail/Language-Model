from vocabulary import Vocabulary
from network import Network
from layer import softmax
from loss import cross_entropy
import numpy as np

with open("tinyshakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()
text = " ".join(text.split()[:500]) #only takes the first 500 words
vocab = Vocabulary(text)

print("Total words:", len(vocab.words))
print("Unique tokens:", len(vocab.vocab))

size = len(vocab.vocab)
network = Network([size, 8, size])

training_pairs = []
for i in range(len(vocab.words) - 1): #creates word pairs
    pair = (vocab.words[i], vocab.words[i+1])
    training_pairs.append(pair)

model_data = {}
model_data["networkShape"] = network.shape
model_data["epochs"] = 0
model_data[f"vocabulary"] = []

for word in vocab.vocab:
    model_data[f"vocabulary"].append(word)

for epoch in range(500):
    total_loss = 0
    for input_word, target_word in training_pairs:
        logits = network.forward(vocab.one_hot(input_word))
        probabilities = softmax(logits)

        loss = cross_entropy(probabilities, vocab.vocab[target_word])
        gradients = probabilities - vocab.one_hot(target_word)

        network.backward(gradients)
        network.update(0.01)
        total_loss += loss

    model_data["epochs"] += 1
    print(f">[{epoch}] Average Loss: {total_loss / len(training_pairs)}")
    if model_data["epochs"] % 250 == 0: #checkpoints every 250 epochs
        for i, layer in enumerate(network.layers):
            model_data[f"weights_{i}"] = layer.weights
            model_data[f"biases_{i}"] = layer.biases
        np.savez("shakespeare.npz", **model_data)

word = "the"
generated = [word]
while word != "END" and len(generated) < 100:
    logits = network.forward(vocab.one_hot(word))
    probabilities = softmax(logits)
    word = vocab.reverseVocab[np.random.choice(len(probabilities), p=probabilities)]
        
    if word == "END": break
    generated.append(word)
print(" ".join(generated)) 