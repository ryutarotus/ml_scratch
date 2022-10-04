import numpy as np
import matplotlib.pyplot as plt

#線形単回帰
class Simple_Linear_Regression():
    def __init__(self):
        self.intercept = None
        self.coef = None
        
    def fit(self, x, y):
        coef = np.cov(x, y)[0][1] / np.var(x)
        intercept = np.mean(y) - coef * np.mean(x)
        
        self.coef = coef
        self.intercept = intercept
        return None
    
    def predict(self, x):
        assert self.intercept != None and self.coef != None, "NotFittedError: Estimator not fitted, please call `fit` before predict the model"
        y = self.coef * x + self.intercept
        return y

#線形重回帰
class Linear_Regression():
    def __init__(self):
        self.intercept = None
        self.coef = None
        
    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X = np.vstack([np.ones(X.shape[1]), X]).T
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        coef = theta[1: ]
        intercept = theta[0]
        self.coef = coef
        self.intercept = intercept
        return None
    
    def predict(self, x):
        assert self.intercept.all() != None and self.coef.all() != None, "NotFittedError: Estimator not fitted, please call `fit` before predict the model"
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        y = np.dot(self.coef, x) + self.intercept
        return y

