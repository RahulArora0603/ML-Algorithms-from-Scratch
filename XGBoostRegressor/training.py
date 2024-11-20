import numpy as np
from xgb_regressor import MyXGBRegressor
import matplotlib.pyplot as plt
                 
# Plot the training data and estimated curve
def plot_prediction(x, y, x_test, y_pred):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c='blue', s=20, alpha=0.5, label='train data')
    plt.plot(x_test, y_pred, c='red', lw=2.0, label='prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 7)
    plt.legend()
    plt.show()

# Generate the training data
def nonlinear_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.random()
       y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)
x, y = nonlinear_data(n=500, s=0.5)

y_mean = y.mean()     # initial prediction
n_depth = 3           # tree depth
n_tree = 20           # the number of trees
eta = 0.3             # learning rate
reg_lambda = 1.0      # regularization constant
prune_gamma = 2.0     # pruning constant

my_model = MyXGBRegressor(n_estimators=n_tree,
                          max_depth=n_depth,
                          learning_rate=eta,
                          prune_gamma=prune_gamma,
                          reg_lambda=reg_lambda,
                          base_score = y_mean)
loss = my_model.fit(x, y)

# Check the loss history
plt.figure(figsize=(5,4))
plt.plot(loss, c='red')
plt.xlabel('m : iteration')
plt.ylabel('loss: mean squared error')
plt.title('loss history')
plt.show()

x_test = np.linspace(0, 1, 50).reshape(-1, 1)
y_pred = my_model.predict(x_test)

# Plot the training data and estimated curve
plot_prediction(x, y, x_test, y_pred)


