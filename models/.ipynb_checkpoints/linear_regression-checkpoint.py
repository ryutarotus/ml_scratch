import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression():
    def __init__(self):
        self.intercept = None
        self.coef = None
        
    def fit(self, x, y):
        x = x.astype('float64')
        y = y.astype('float64')
        coef = np.cov(x, y)[0][1] / np.var(x)
        intercept = np.mean(y) - coef * np.mean(x)
        
        self.coef = coef
        self.intercept = intercept
        return None
    
    def predict(self, x):
        if self.intercept == None or self.coef == None:
            print('Please fit model !')
            return 'process is interruputed'
        else:
            x = x.astype('float64')
            y = self.coef * x + self.intercept
        return y
    
    def plot_result(self, x, y, pred):
        plt.scatter(x, y)
        plt.plot(x, pred)
        plt.show()