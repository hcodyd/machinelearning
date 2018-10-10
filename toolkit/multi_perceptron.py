from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np


class MultiLearner(SupervisedLearner):
    """
        For nominal labels, this model simply returns the majority class. For
        continuous labels, it returns the mean value.
        If the learning model you're using doesn't do as well as this one,
        it's time to find a new learning model.
    """
    labels = []
    biasWeight = []
    learningRate = .1
    epics = 5



    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.weights = np.zeros((len(labels.str_to_enum[0]), features.cols))
        self.bias = -1
        self.biasWeight = np.zeros((len(labels.str_to_enum[0])))
        for k in range(self.epics):
            features.shuffle(labels)
            for i in range(features.rows):
                for j in range(len(self.weights)):
                    mytarget = 0
                    if j == labels.get(i, 0):
                        mytarget = 1
                    row = features.row(i)
                    # row.append(self.bias)
                    bias_input = np.sum(self.bias * self.biasWeight[j])
                    net = np.sum(row * self.weights[j] + bias_input)
                    output = 1 if net > 0 else 0
                    if output != mytarget:
                        delta = []
                        for l in range(0, len(self.weights[j])):
                            delta.append(self.learningRate * (mytarget - output) * features.get(i, l))
                        self.weights[j] = self.weights[j] + delta
                        self.biasWeight[j] = self.learningRate * (mytarget - output) * self.bias



    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        print(self.weights.view())
        del labels[:]
        # labels += self.labels
        temp = []
        for i in range(len(self.weights)):
            bias_input = np.sum(self.bias * self.biasWeight[i])
            net = np.sum(features * self.weights[i] + bias_input)
            print(net)
            # output = 1 if net > 0 else 0
            temp.append(net)

        print(temp)
        labels.append(temp.index(max(temp)))

