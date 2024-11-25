""" Prototype """

#import warnings
#import random

import numpy as np
#import dace as dc

# phase out
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

# For vis
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



class GaussianMixtureModel():
    """ Gaussian Mxiture Model class """
    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.
    def __init__(self, data, nr_of_components):
        self.data = data

        #self.samles_, self.number_features = data.shape
        _, self.nr_of_features = data.shape
        self.nr_of_components = nr_of_components

        # Why pi, what is pi in case of multibariate algorithm
        self.mean_, \
            self.covariance_, \
            self.pi_ = self.random_init()

        self.mean_history = []
        self.covariance_history = []

    def random_init(self) -> tuple([np.array, np.array, np.array]):
        """ Should have different inits """

        pi_ = np.ones((self.nr_of_components)) / self.nr_of_components
        # Should probably store length of data form begining (from shape)
        mean_ = self.data[np.random.choice(len(self.data[:,0]),
                                           self.nr_of_components,
                                           replace=False)]
        covariance_ = [np.eye(self.nr_of_features)] * self.nr_of_components
        return mean_, covariance_, pi_

    def warmstart_init(self):
        """ Init with some guesses """

        return 0

    def set_mean(self, mean : np.array) -> None:
        """ Set mean before or after training """ 
        self.mean_ = mean

    def expectation(self) -> []:
        """ Calculate the expectation """
        weights = []
        for k in range(self.nr_of_components):
            numerator = multivariate_normal.pdf(self.data,
                                                mean=self.mean_[k],
                                                cov=self.covariance_[k]) * self.pi_[k]
            weights.append(numerator)
        weights = np.array(weights).T
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def maximization(self, weights: np.array) -> None:
        """ Calculate maximization
        Args:
            weights (np.array)
        """
        # This calculation is deeply unstable numerically
        # WARNING

        total_resp = np.sum(weights, axis=0)
        self.pi_ = total_resp / 800 #self.nr_of_components
        self.mean_ = (weights.T @ self.data) / total_resp[:, np.newaxis]
        for k in range(self.nr_of_components):
            diff = self.data - self.mean_[k]
            self.covariance_[k] = ((weights[:, k] * diff.T) @ diff) / total_resp[k]
            # Covariancve must be SPD lets find a better way
            self.covariance_[k] = 0.5*(self.covariance_[k].T+self.covariance_[k]) \
                    +np.eye(self.nr_of_features)*1e-10

    def train(self, nr_of_steps: int = 50) -> None:
        """ Train the model
        Arg:
            number_steps (int)
        """

        for _ in range(nr_of_steps):
            weights = self.expectation()
            self.maximization(weights)
            self.mean_history.append(self.mean_.copy())
            self.covariance_history.append(self.covariance_.copy())

            # some intelligent test for
            # convergence

            # some inteligent handling of very small
            # variance
        print('Training finished')

    def evaluate(self, component: int, nr_of_samples: int) -> None:
        """ draw from distribution """
        #return np.random.normal(loc = self.mean_[component],
        #                     scale = np.sqrt(self.covariance_[component]),
        #                     size = nr_of_samples)
        return np.random.multivariate_normal(mean=self.mean_[component],
                                   cov=self.covariance_[component],
                                   size=(nr_of_samples,)
                                   )


    def evaluate_equal(self, nr_of_samples_tot: int) -> np.array:
        """ draw eq amount of samples from all distirbutions
        make this nicer lol """
        predicted_data = np.zeros((nr_of_samples_tot,self.nr_of_features))
        component_size = np.ndarray(shape=(self.nr_of_components,),dtype=np.int32)
        component_size[:] = int(np.floor_divide(nr_of_samples_tot,self.nr_of_components))
        component_size[0] += int(nr_of_samples_tot%self.nr_of_components)
        stop = 0
        for component in range(self.nr_of_components):
            start = stop
            stop += int(component_size[component])
            predicted_data[start:stop,:] = self.evaluate(component,int(component_size[component]))

        return predicted_data

    def inspect(self):
        """ Viz or something """
        for i in range(self.nr_of_components):
            print("mean: ", self.mean_[i], " variance: ", (self.covariance_[i]).flatten())


    def get_mean(self) -> np.array:
        """ returns mean only """
        return self.mean_

    def get_history(self) -> (list, list):
        """ returns the history of the training """
        return self.mean_history, self.covariance_history

def test_1d(nr_of_data_points: int, plot_fig: bool):
    """ Test in 1D """
    mu1, sigma1 = -5,1.2
    mu2, sigma2 = 5, 6
    mu3, sigma3 = 0, 1.6

    x1 = np.random.normal(loc = mu1, scale = np.sqrt(sigma1), size = nr_of_data_points)
    x2 = np.random.normal(loc = mu2, scale = np.sqrt(sigma2), size = nr_of_data_points)
    x3 = np.random.normal(loc = mu3, scale = np.sqrt(sigma3), size = nr_of_data_points)

    x = np.concatenate((x1,x2,x3))

    print('----')
    print('Test on distribution built on')
    print("mu1 ",mu1," sigma1 ", sigma1)
    print("mu2 ",mu2," sigma2 ", sigma2)
    print("mu3 ",mu3," sigma3 ", sigma3)
    print('----')

    model = GaussianMixtureModel(x.reshape(-1,1),3)
    #model.set_mean(np.array([-1,0,1]))
    model.inspect()

    mean, _ = model.get_history()
    data = model.evaluate_equal(3*nr_of_data_points)
    sns.set_theme()
    if plot_fig:
        plt.figure()
        temp_dataframe = pd.DataFrame({'GMM' : data[:,0], 'Data' : x})
        sns.histplot(temp_dataframe, bins=100, kde=True)
        plt.savefig('hist1D'+str(0)+'.svg')
        plt.show()

    for step in range(1,3):
        model.train(nr_of_steps = 30)
        model.inspect()
        mean, _ = model.get_history()
        data = model.evaluate_equal(3*nr_of_data_points)
        if plot_fig:
            sns.set_theme()
            mean = (np.array(mean)).flatten()
            sns.relplot(pd.DataFrame(mean))
            plt.show()
            temp_dataframe = pd.DataFrame({'GMM' : data[:,0], 'Data' : x})
            plt.figure()
            sns.histplot(temp_dataframe, bins=100, kde=True)
            plt.savefig('hist1D'+str(step*15)+'.svg')
            plt.show()

def test_2d(nr_of_data_points: int,
            nr_of_features: int,
            nr_of_centers: int) -> None:
    """ Including plotting """
    data, _ = make_blobs(n_samples = nr_of_data_points,
                                   n_features = nr_of_features,
                                   centers = nr_of_centers,
                                   random_state = 10)
    model = GaussianMixtureModel(data, nr_of_components = nr_of_centers)
    model.train(nr_of_steps = 80)
    mean_x, mean_y = model.get_mean().T
    pos, vel = model.evaluate_equal(nr_of_data_points).T

    sns.scatterplot(x=data[:,0], y=data[:,1])
    sns.scatterplot(x=mean_x, y=mean_y, color='r')
    sns.kdeplot(x=data[:,0], y=data[:,1], levels=5, color="k", linewidths=1)
    sns.kdeplot(x=pos, y=vel, levels=5, color="y", linewidths=1)
    plt.show()

def test_with_moons(nr_of_data_points: int,
                        nr_of_components: int) -> None:
    """ test_with_moons """
    data, _ = make_moons(nr_of_data_points, noise=.05, random_state=10)

    model = GaussianMixtureModel(data, nr_of_components = nr_of_components)
    model.train(nr_of_steps = 1000)
    mean_x, mean_y = model.get_mean().T
    model.inspect()
    pos, vel = model.evaluate_equal(nr_of_data_points).T

    sns.scatterplot(x=data[:,0], y=data[:,1])
    sns.scatterplot(x=mean_x, y=mean_y, color='r')
    sns.kdeplot(x=data[:,0], y=data[:,1], levels=3, color="k", linewidths=1)
    sns.kdeplot(x=pos, y=vel, levels=3, color="g", linewidths=1)
    plt.xlabel('x')
    plt.ylabel('v')
    plt.show()



if __name__ == "__main__":
    test_1d(nr_of_data_points = 800, plot_fig=True)
    test_2d(nr_of_data_points = 400, nr_of_features = 2, nr_of_centers = 3)
    test_with_moons(nr_of_data_points = 400, nr_of_components = 2)
    test_with_moons(nr_of_data_points = 400, nr_of_components = 8)
    test_with_moons(nr_of_data_points = 400, nr_of_components = 12)
    test_with_moons(nr_of_data_points = 400, nr_of_components = 16)
