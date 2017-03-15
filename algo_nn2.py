# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:34:27 2015

@author: Ziyan Wang
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


class BackPropagationNetwork:
    #
    # Class Members
    #
    layerCount = 0
    shape = None
    weights = []

    #
    # Class methods
    #
    def __init__(self, layerSize):
        """ Initialize the network """
        # layer size(input layer size, hidden layer 1 size, hidden layer 2 size..., output layer size)

        # layer information
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        # data from last run
        self._layerInput = []
        self._layerOutput = []

        # Create the weight arrays
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):  # connection
            self.weights.append(np.random.normal(scale=0.01, size=(l2, l1 + 1)))  # one more weight for bias term

    #
    # Run Method
    #    
    def Run(self, inputData):
        """ Run the network with input data """

        lnCases = inputData.shape[0]

        # Clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []

        # Main part
        for index in range(self.layerCount):  # go through each layer
            # determine layer input
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([inputData.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T

    def TrainEpoch(self, inputData, target, trainingRate=0.2):
        """ trains the network for one epoch """

        delta = []
        lnCases = inputData.shape[0]

        self.Run(inputData)

        # calculate deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                # compare to the target values
                outputDelta = self._layerOutput[index] - target.T
                error = np.sum(outputDelta ** 2)
                delta.append(outputDelta * self.sgm(self._layerInput[index], True))
            else:
                # compare to the following layer's delta
                deltaPullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(deltaPullback[:-1, :] * self.sgm(self._layerInput[index], True))

        # compute weight deltas
        for index in range(self.layerCount):
            deltaIndex = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([inputData.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            weightDelta = np.sum(
                layerOutput[None, :, :].transpose(2, 0, 1) * delta[deltaIndex][None, :, :].transpose(2, 1, 0)
                , axis=0)

            self.weights[index] -= trainingRate * weightDelta

        return error

    def sgm(self, x, Derivative=False):
        """ Transfer function: sigmoid"""

        if not Derivative:
            return 1 / (1 - np.exp(-x))
        else:
            out = self.sgm(x)
            return out * (1 - out)


if __name__ == '__main__':
    """ Load data """
    iris=load_iris()
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data[:,:2], iris.target, test_size=0.3, random_state=11)

    """ Assign in sample and out of sample data """

    lvInput = X_train
    lvTarget = y_train

    """ Train the network """
    lnMax = 3000
    lnErr = 1e-2

    # build up neural network structure
    bpn = BackPropagationNetwork((X_train.shape[1], 10, 1))
    # print(bpn.shape)
    # print(bpn.weights)

    for i in range(lnMax - 1):
        err = bpn.TrainEpoch(lvInput, lvTarget, 0.05)
        if i % 1000 == 0:
            print("Iteration {0}\tError: {1:0.6f}".format(i, err))
        if err <= lnErr:
            print("Minimun error reached at iteration {0}".format(i))
            break

    """ display and compare outputs """
    lvOutput = bpn.Run(lvInput)
    plt.scatter([i for i in range(len(lvOutput))], lvOutput, color='blue')
    plt.scatter([i for i in range(len(lvTarget))], lvTarget, color='black')

    plt.xticks(())
    plt.yticks(())

    plt.show()
