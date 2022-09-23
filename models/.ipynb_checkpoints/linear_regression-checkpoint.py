import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression():
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
    """
    def plot_result(self, x, y, pred):
        plt.scatter(x, y)
        plt.plot(x, pred)
        plt.show()
    """