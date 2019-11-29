# coding: utf-8
import math
import numpy as np
from sklearn.datasets import load_iris
from collections import Counter

from sklearn.model_selection import train_test_split


class NavieBayes(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def mean(x):
        return sum(x) / len(x)

    def stdev(self, x):
        avg = self.mean(x)
        stdev = (math.sqrt(sum([(i - avg) ** 2 for i in x]))) / len(x)
        return stdev

    # 处理训练数据
    def summarize(self, X_train):
        result = [(self.mean(i), self.stdev(i)) for i in zip(*X_train)]
        return result

    # 计算概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }

    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    def predict(self, X_test):
        label = sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))


if __name__ == '__main__':
    data = load_iris()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1)
    clf = NavieBayes()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
