import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# log(odds) --> probability
def F2P(f):
    return 1. / (1. + np.exp(-f))

# Creating training data
x, y = make_blobs(n_samples=200, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.5]], 
                  cluster_std=0.18, center_box=(-1., 1.))
       
n_data = x.shape[0]
n_depth = 2    # tree depth
n_tree = 50    # the number of trees (M)
alpha = 0.1    # learning rate

# step-1: Initialize model with a constant value.
F0 = np.log(y.mean() / (1. - y.mean()))
Fm = np.repeat(F0, n_data)

models = []
loss = []
for m in range(n_tree):
    # step-2 (A): Compute so-called pseudo-residuals
    y_hat = F2P(Fm)
    residual = y - y_hat
    
    # step-2 (B): Fit a regression tree to the residual
    gb_model = DecisionTreeRegressor(max_depth=n_depth)
    gb_model.fit(x, residual)
    
    # The leaf nodes of this tree contain the average of the 
    # residuals. The predict() function returns this average value. 
    # We replace these values ​​with the leaf values gamma. Then the 
    # predict() function will return the gamma.
    
    # step-2 (C): compute gamma
    # leaf_id = The leaf node number to which x belongs.
    leaf_id = gb_model.tree_.apply(x.astype(np.float32))
    
    # Replace the leaf values ​​of all leaf nodes with their gamma 
    # values, ​​and update Fm. 
    for j in np.unique(leaf_id):
        # i=Index of data points belonging to leaf node j.
        i = np.where(leaf_id == j)[0]
        gamma = residual[i].sum() / (y_hat[i] * (1. - y_hat[i])).sum()

        # step-2 (D): Update the model
        Fm[i] += alpha * gamma
        
        # Replace the leaf values ​​with their gamma
        # gb_model.tree_.value.shape = (7, 1, 1)
        gb_model.tree_.value[j, 0, 0] = gamma

    # save the trained model
    models.append(gb_model)
    
    # Calculating loss. loss = binary cross entropy.
    loss.append(-(y * np.log(y_hat + 1e-8) + \
                 (1.- y) * np.log(1.- y_hat + 1e-8)).sum())

# Check the loss history visually.
plt.figure(figsize=(5,4))
plt.plot(loss, c='red')
plt.xlabel('m : iteration')
plt.ylabel('loss: binary cross entropy')
plt.title('loss history')
plt.show()

# step-3: Output Fm(x) - Prediction of test data
Fm = F0
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))

for model in models:
    Fm += alpha * model.predict(x_test)
    
y_prob = F2P(Fm)
y_pred = (y_prob > 0.5).astype('uint8')

# Visualize training and prediction results.
def plot_prediction(x, y, x_test, y_pred):
    plt.figure(figsize=(5,5))
    color = ['red' if a == 1 else 'blue' for a in y_pred]
    plt.scatter(x_test[:, 0], x_test[:, 1], s=100, c=color, 
                alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], s=80, c='black')
    plt.scatter(x[:, 0], x[:, 1], s=10, c='yellow')
    plt.xlim(-0.5, 1.0)    
    plt.ylim(-0.5, 1.0)
    plt.show()
    
# Visualize test data and y_pred.
plot_prediction(x, y, x_test, y_pred)

