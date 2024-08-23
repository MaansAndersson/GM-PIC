""" Prototype """

import numpy as np

# Phase out
from scipy.stats import multivariate_normal
from scipy.stats import norm

import warnings
import random

import dace as dc
import numpy as np

class GaussianMixtureModel(object):

    def __init__(self, data, number_components):
        self.data = data
        self.number_components = number_components
        self.mean_, self.variance_, self.pi_ = self.random_init() 

    def random_init(self):
        pi_ = np.ones((self.number_components))/self.number_components
        mean_ = np.random.choice(self.data, self.number_components)
        variance_ = np.random.random_sample(size=self.number_components)
        
        return mean_, variance_, pi_
    
    def expectation(self):
        weights = np.zeros((self.number_components,len(self.data)))
        for j in range(self.number_components):
            weights[j,:] = norm(loc=self.mean_[j],scale=np.sqrt(self.variance_[j])).pdf(self.data)
        return weights
    
    def maximization(self, weights : np.array):
        r = []

        # Rewrite with nice parallel map
        for j in range(self.number_components):
            r.append((weights[j] * self.pi_[j]) / (np.sum([weights[i] * self.pi_[i] for i in range(self.number_components)],axis=0))) 
            self.mean_[j] = np.sum(r[j] * self.data)/(np.sum(r[j]))
            self.variance_[j] = np.sum(r[j] * np.square(self.data - self.mean_[j])) / np.sum(r[j])
            self.pi_[j] = np.mean(r[j])
     
    def train(self, number_steps=150, plot_intermediate_step_flag=True):
        self.mean_history, self.variance_history = [], []
        for step in range(number_steps):
            weights = self.expectation()
            self.maximization(weights)
            
            self.mean_history.append(self.mean_.copy())
            self.variance_history.append(self.variance_.copy())

            # some intelligent test for
            # convergence
            
            # some

    """ draw from distribution """
    def evaluate(self, component, n_samples):
        return np.random.normal(loc = self.mean_[component],
                             scale = np.sqrt(self.variance_[component]),
                             size = n_samples)        

    """ Viz or something """
    def instpect(self):
        for i in range(self.number_components):
            print("mean: ", self.mean_[i], " variance: ", self.variance_[i])

    def get_history(self):
        return self.mean_history, self.variance_history



if __name__ == "__main__":
    n_samples = 100
    mu1, sigma1 = -5, 1.2
    mu2, sigma2 = 5, 1.8
    mu3, sigma3 = 0, 1.6
    mu4, sigma4 = 3, 2
    
    x1 = np.random.normal(loc = mu1, scale = np.sqrt(sigma1), size = n_samples)
    x2 = np.random.normal(loc = mu2, scale = np.sqrt(sigma2), size = n_samples)
    x3 = np.random.normal(loc = mu3, scale = np.sqrt(sigma3), size = n_samples)
    x4 = np.random.normal(loc = mu4, scale = np.sqrt(sigma4), size = n_samples)
   
    X = np.concatenate((x1,x2,x3,x4))
   
    print('----')
    print('Test on distribution built on')
    print("mu1 ",mu1," sigma1 ", sigma1)
    print("mu2 ",mu2," sigma2 ", sigma2)
    print("mu3 ",mu3," sigma3 ", sigma3)
    print("mu4 ",mu4," sigma4 ", sigma4)
    print('----')
    model = GaussianMixtureModel(X,3)
    model.train()
    model.instpect()
   
    mean, variance = model.get_history()
    
    import seaborn as sns
    import pandas as pd
    sns.set_theme()

    sns.relplot(pd.DataFrame(mean))
   
    import matplotlib.pyplot as plt
    plt.show()


