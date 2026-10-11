import numpy as np

class Vocabulary:
    def __init__(self, text):
        self.words = text.split()
        self.words.append("END")

        self.vocab = {}
        for word in self.words: #creates a dictionary of all the words with unique ids
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        self.reverseVocab = {}
        for word in self.vocab:
            self.reverseVocab[self.vocab[word]] = word

    def encode(self, text):
        words = text.split()
        ids = []
        for word in words:
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids):
        words = []
        for token_id in ids:
            words.append(self.reverseVocab[token_id])
        return " ".join(words)

    def one_hot(self, word):
        vector = [0] * len(self.vocab)
        vector[self.vocab[word]] = 1
        return np.array(vector)