from decision_trees import DecisionTree #import Decision Tree 
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees #number of decision trees in the forest
        self.max_depth=max_depth #maximum depth of each decision tree
        self.min_samples_split=min_samples_split #Min number of samples required to split a node
        self.n_features=n_feature #Number of features to consider when spitting
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees): #to create 'n_trees' no of individual trees
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y): #generates a bootstrap sample of the data
        n_samples = X.shape[0] 
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y): #find the most common label in array
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X): #predicts class labels by aggregating predictions from all decision trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions