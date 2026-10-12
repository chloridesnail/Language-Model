import numpy as np

class Vocabulary:
    def __init__(self, text: str):
        self.chars = list(text)

        self.vocab = {}
        for char in self.chars: #creates a dictionary of all the chars with unique ids
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        self.reverseVocab = {}
        for char in self.vocab:
            self.reverseVocab[self.vocab[char]] = char

    def encode(self, text):
        chars = list(text)
        ids = []
        for char in chars:
            ids.append(self.vocab[char])
        return ids

    def decode(self, ids):
        chars = []
        for token_id in ids:
            chars.append(self.reverseVocab[token_id])
        return "".join(chars)

    def one_hot(self, char):
        vector = [0] * len(self.vocab)
        vector[self.vocab[char]] = 1
        return np.array(vector)
    
    def one_hotv2(self, chars):
        array = []
        for char in chars:
            vector = self.one_hot(char)
            array.append(vector)
        return np.concatenate(array)

    def load(self, saved_vocab):
        self.vocab = {}
        self.reverseVocab = {}
        for i, char in enumerate(saved_vocab):
            self.vocab[char] = i
            self.reverseVocab[i] = char