import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 
from linear_from_scratch import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features = 1, noise=20, random_state=42) #create random data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #split into train and test


reg = LinearRegression(lr=0.01)  #create Linear Regression type object
reg.fit(X_train, y_train)  #train the data
predictions = reg.predict(X_test) #predict

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2) #mean squared error

mse = mse(y_test,predictions)
print(mse)

#Plotting the model
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train , color=cmap(0.9),s=10)
m2 = plt.scatter(X_test, y_test , color=cmap(0.5),s=10)
plt.plot(X, y_pred_line, color="black",linewidth = 1, label='Prediction')
plt.show()