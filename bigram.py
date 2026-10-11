import random
from typing import List
import numpy as np

text = "the cat eats the cat sleeps the cat naps"
words = text.split()
words.append("END")

pairs = []
counts = {}
totals = {}
probabilities = {}

vectors = {}
vocab = {}

for i in range(len(words) - 1): #creates word pairs
    pair = (words[i], words[i+1])
    pairs.append(pair)

for word in words: #creates a dictionary of all the words with unique ids
    if word not in vocab:
        vocab[word] = len(vocab)

for word in vocab: #hot encodes all the words in vocab
    vector = [0] * len(vocab)
    vector[vocab[word]] = 1
    vectors[word] = vector

for pair in pairs: #counts how many times a pair exists
    if pair in counts:
        counts[pair] += 1
    else:
        counts[pair] = 1
    if pair[0] in totals: #counts how many times a word appears as the first word in a pair
        totals[pair[0]] += 1
    else:
        totals[pair[0]] = 1

for pair in counts: #calculates the probablities using: probability = how many times the pair exists / how many pairs start with the word
    probabilities[pair] = counts[pair] / totals[pair[0]]
    
word = input("Enter a word: ")
generated = [word]

while word != "END" and len(generated) < 20: #creates the sentence
    number = random.random()
    cumulative = 0
    prediction = ""

    for p in probabilities:
        if p[0] == word:
            cumulative += probabilities[p]
            if number < cumulative:
                prediction = p[1]
                if prediction != "END":
                    generated.append(prediction)
                break
    if prediction == "":
        break
    word = prediction