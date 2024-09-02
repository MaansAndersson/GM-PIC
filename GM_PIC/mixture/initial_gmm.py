""" Prototype """

#import warnings
#import random

import numpy as np
#import dace as dc

# Phase out
#from scipy.stats import multivariate_normal
from scipy.stats import norm




class GaussianMixtureModel():
    """ Gaussian Mxiture Model class """
    def __init__(self, data, number_components):
        self.data = data
        self.number_components = number_components
        self.mean_, self.variance_, self.pi_ = self.random_init()

        self.mean_history = []
        self.variance_history = []

    def random_init(self):
        """ Should have different inits """
        pi_ = np.ones((self.number_components))/self.number_components
        mean_ = np.linspace(-4,4,self.number_components)
        #np.random.choice(self.data, self.number_components)
        variance_ = np.ones((self.number_components))
        #self.number_components #np.random.random_sample(size=self.number_components)
        return mean_, variance_, pi_

    def expectation(self):
        """ Calculate the expectation """
        weights = np.zeros((self.number_components,len(self.data)))
        for j in range(self.number_components):
            weights[j,:] = norm(loc=self.mean_[j],scale=np.sqrt(self.variance_[j])).pdf(self.data)
        return weights

    def maximization(self, weights : np.array):
        """ Calculate maximization
        Args:
            weights (np.array)
        """
        r = []

        # Rewrite with nice parallel map
        for j in range(self.number_components):
            r.append((weights[j] * self.pi_[j]) / (np.sum([weights[i] * self.pi_[i] for i in range(self.number_components)],axis=0)))
            self.mean_[j] = np.sum(r[j] * self.data)/(np.sum(r[j]))
            # note stupid handling of small variance
            self.variance_[j] = np.max([np.sum(r[j] * np.square(self.data - self.mean_[j])) / np.sum(r[j]),1e-20])
            self.pi_[j] = np.mean(r[j])

    def train(self, number_steps=50) -> None:
        """ Train the model
        Arg:
            number_steps (int)
        """

        for _ in range(number_steps):
            weights = self.expectation()
            self.maximization(weights)
            self.mean_history.append(self.mean_.copy())
            self.variance_history.append(self.variance_.copy())

            # some intelligent test for
            # convergence

            # some inteligent handling of very small
            # variance

    def evaluate(self, component, n_samples : int):
        """ draw from distribution """
        return np.random.normal(loc = self.mean_[component],
                             scale = np.sqrt(self.variance_[component]),
                             size = n_samples)
    def evaluate_equal(self, n_samples_tot : int):
        """ draw eq amount of samples from all distirbutions 
        make this nicer lol """
        
        predicted_data = np.ndarray((0,))
        component_size = np.ndarray(shape=(self.number_components,),dtype=np.int32)
        component_size[:] = int(np.floor_divide(n_samples_tot,self.number_components))
        component_size[0] += int(n_samples_tot%self.number_components)
        for component in range(self.number_components):
            predicted_data = np.append(predicted_data,self.evaluate(component,int(component_size[component])))
        return predicted_data[:]

    def inspect(self):
        """ Viz or something """
        for i in range(self.number_components):
            print("mean: ", self.mean_[i], " variance: ", self.variance_[i])

    def get_history(self):
        """ returns the history of the training """
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
    model.inspect()
    mean, variance = model.get_history()
    import seaborn as sns
    import pandas as pd
    sns.set_theme()

    sns.relplot(pd.DataFrame(mean))
    import matplotlib.pyplot as plt
    plt.show()
