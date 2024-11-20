import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr   #learning rate
        self.n_iters = n_iters #number of iterations
        self.weights = None #weights
        self.bias = None   #bias

    def fit(self, X, y):
        n_samples, n_features = X.shape #n_samples - no. of rows, n_features - number of columns
        self.weights = np.zeros(n_features) #array of length n_features where each weight is 0
        self.bias = 0 

        for _ in range(self.n_iters): #loop
            linear_pred = np.dot(X, self.weights) + self.bias #dot product of data and weights + bias
            predictions = sigmoid(linear_pred) #apply sigmoid function to data

            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) #derivative of weights
            db = (1/n_samples) * np.sum(predictions-y) #derivative of bias

            self.weights = self.weights - self.lr*dw #updating the weights
            self.bias = self.bias - self.lr*db #updating the bias


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias #dot product based on final weights and bias
        y_pred = sigmoid(linear_pred) #final prediction
        class_pred = [0 if y<=0.5 else 1 for y in y_pred] #classifying
        return class_pred

    