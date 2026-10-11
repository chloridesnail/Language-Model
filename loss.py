import numpy as np

def cross_entropy(probabilities, target_id):
    return -(np.log(probabilities[target_id]))