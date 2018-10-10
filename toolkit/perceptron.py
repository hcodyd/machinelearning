from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np


class PerceptronLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """
    learningRate = .1
    labels = []
    epochSum = 1
    biasWeight = 0
    trueVal = True
    epochs = 0

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.bias = -1
        self.weights = np.zeros(features.cols)
        self.labels = []
        while self.trueVal:
            features.shuffle(labels)
            self.epochs += 1
            for j in range(features.rows):
                row = features.row(j)
                bias_input = np.sum(self.bias * self.biasWeight)
                net = np.sum(row * self.weights + bias_input)
                output = 1 if net > 0 else 0
                if output != labels.row(j):
                    delta = []
                    for l in range(0, len(self.weights)):
                        delta.append(self.learningRate*(labels.get(j, 0) - output)*features.get(j, l))
                    self.weights = self.weights + delta
                    self.biasWeight = self.learningRate*(labels.get(j, 0)-output)*self.bias

                if abs((self.epochSum - np.sum(self.weights)))/((self.epochSum+np.sum(self.weights))/2) > .1:
                    self.epochSum = np.sum(self.weights)
                else:
                    self.trueVal = False



    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        # print("features", features)
        # print("weights", self.weights)
        # print("labels", labels)

        del labels[:]
        row = features
        # row.append(self.bias)
        net = np.sum(row * self.weights)
        # print(self.weights, self.biasWeight)
        output = 1 if net > 0 else 0
        labels.append(output)
        # labels += self.labels

    #
    # def measure_accuracy(self, features, labels, confusion=None):
    #     super(PerceptronLearner, self).measure_accuracy(features, labels, confusion)

