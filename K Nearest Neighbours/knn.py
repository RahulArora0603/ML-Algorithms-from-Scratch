import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2)) #euclidean distance
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k #number of neighbours

    def fit(self, X, y): #lazy training
        self.X_train = X 
        self.y_train = y

    def predict(self, X): #prediction
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k (smallest value)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0] #returns label with highest count 