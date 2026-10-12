from vocabulary import Vocabulary
from network import Network
from layer import softmax
from loss import cross_entropy
import numpy as np

epochs = 2000
def train(network, vocab, current_epoch):

    training_octs = []
    for i in range(len(vocab.chars) - 32): #creates char octs
        oct = []
        zero = []
        for v in range(32):
            zero.append(vocab.chars[i + v])
        oct.append(zero)
        oct.append(vocab.chars[i + 32])
        training_octs.append(tuple(oct))

    model_data = {}
    model_data["networkShape"] = network.shape
    model_data["epochs"] = current_epoch
    model_data[f"vocabulary"] = []

    for char in vocab.vocab:
        model_data[f"vocabulary"].append(char)

    for epoch in range(epochs):
        total_loss = 0
        for input_char, target_char in training_octs:
            logits = network.forward(vocab.one_hotv2(input_char))
            probabilities = softmax(logits)

            loss = cross_entropy(probabilities, vocab.vocab[target_char])
            gradients = probabilities - vocab.one_hot(target_char)

            network.backward(gradients)
            network.update(0.01)
            total_loss += loss

        model_data["epochs"] += 1
        print(f">[{model_data['epochs']}] Average Loss: {total_loss / len(training_octs)}")
        if model_data["epochs"] % 5 == 0: #checkpoints every 5 epochs
            for i, layer in enumerate(network.layers):
                model_data[f"weights_{i}"] = layer.weights
                model_data[f"biases_{i}"] = layer.biases
            np.savez("shakespeare_char_32ctx.npz", **model_data)

    for i, layer in enumerate(network.layers):
        model_data[f"weights_{i}"] = layer.weights
        model_data[f"biases_{i}"] = layer.biases
    np.savez("shakespeare_char_32ctx.npz", **model_data)

training_chars = 100000
mode = "generate"
if mode == "new":
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = "".join(list(text)[:training_chars])
    vocab = Vocabulary(text)
    network = Network([len(vocab.vocab) * 32, 64, len(vocab.vocab)])
    train(network, vocab, 0)
elif mode == "generate":
    save_data = np.load("shakespeare_char_32ctx.npz")
    network = Network(save_data["networkShape"])
    vocab = Vocabulary("")

    network.load(save_data)
    vocab.load(save_data["vocabulary"])
elif mode == "resume":
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = "".join(list(text)[:training_chars])
    save_data = np.load("shakespeare_char_32ctx.npz")

    vocab = Vocabulary(text)
    network = Network(save_data["networkShape"])
    current_epoch = save_data["epochs"]
    
    network.load(save_data)
    vocab.load(save_data["vocabulary"])
    save_data.close()
    train(network, vocab, current_epoch)   

context = "First Citizen:\nBefore we proceed"
generated = f"{context}"
while len(generated) < 1000:
    logits = network.forward(vocab.one_hotv2(context))
    probabilities = softmax(logits)
    next_char = vocab.reverseVocab[np.random.choice(len(probabilities), p=probabilities)]
    context = f"{context[1:]}{next_char}"

    generated = f"{generated}{next_char}"
print(generated) 