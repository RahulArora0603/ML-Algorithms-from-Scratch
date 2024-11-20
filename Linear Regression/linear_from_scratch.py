import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr   #learning rate
        self.n_iters = n_iters #number of iterations
        self.weights = None   #weights
        self.bias = None      #bias

    def fit(self, X, y):         #fit method (training happens here)
        n_samples, n_features = X.shape #n_samples --> number of rows, n_features --> number of cols
        self.weights = np.zeros(n_features) #array of length n_features filled with zeros
        self.bias = 0  

        for _ in range(self.n_iters):  #loop

            y_pred = np.dot(X, self.weights) + self.bias  #dot product of data and weights + bias (mx+b)
    
            dw = (1/n_samples)*np.dot(X.T, (y_pred-y)) #derivative of weights
            db = (1/n_samples)*np.sum(y_pred-y)        #derivative of bias
    
            self.weights = self.weights - self.lr * dw  #updating the weights using dw and learning rate
            self.bias = self.bias - self.lr * db        # "    "      bias     " "  db and learning rate

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias #final prediction
        return y_pred
        