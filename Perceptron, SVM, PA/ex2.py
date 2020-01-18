import sys
import numpy as np


class Perceptron(object):
    def __init__(self, learningRate=0.01, iterationsAmount=10):
        self.learningRate = learningRate
        self.iterationsAmount = iterationsAmount

    def fit(self, train_x, train_y):
        numberOfClasses = np.unique(train_y).size
        weights = np.zeros((numberOfClasses, train_x[0].size))

        for _ in range(self.iterationsAmount):
            for xi, yi in zip(train_x, train_y):
                y_hat = np.argmax(np.dot(weights, xi))

                if(yi != y_hat):
                    weights[yi, :] += self.learningRate * xi
                    weights[y_hat, :] -= self.learningRate * xi

        return weights


class SupportVectorMachine(object):
    def __init__(self, learningRate=0.01, iterationsAmount=10, lamda=0.1):
        self.lamda = lamda
        self.learningRate = learningRate
        self.iterationsAmount = iterationsAmount

    def fit(self, train_x, train_y):
        numberOfClasses = np.unique(train_y).size
        weights = np.zeros((numberOfClasses, train_x[0].size))

        for _ in range(self.iterationsAmount):
            for xi, yi in zip(train_x, train_y):
                y_hat = np.argmax(np.dot(weights, xi))

                if(yi != y_hat):
                    beta = 1 - self.learningRate * self.lamda
                    weights[yi, :] = beta * weights[yi, :] + self.learningRate * xi
                    weights[y_hat, :] = beta * weights[y_hat, :] - self.learningRate * xi

                    otherWeights = np.arange(0, len(weights)).tolist()
                    otherWeights.remove(yi)
                    otherWeights.remove(y_hat)
                    weights[otherWeights, :] = (1 - self.learningRate * self.lamda) * weights[otherWeights, :]
                else:
                    weights[:, :] = (1 - self.learningRate * self.lamda) * weights[:, :]

        return weights


class PassiveAggressive(object):
    def __init__(self, iterationsAmount=10):
        self.iterationsAmount = iterationsAmount

    def fit(self, train_x, train_y):
        numberOfClasses = np.unique(train_y).size
        weights = np.zeros((numberOfClasses, train_x[0].size))

        for _ in range(self.iterationsAmount):
            for xi, yi in zip(train_x, train_y):
                y_hat = np.argmax(np.dot(weights, xi))

                if(yi != y_hat):
                    loss = max(0, 1 - np.dot(weights[yi, :], xi) + np.dot(weights[y_hat, :], xi))
                    gama = loss / (2 * (np.linalg.norm(xi) ** 2))

                    weights[yi, :] += gama * xi
                    weights[y_hat, :] -= gama * xi

        return weights


def prepareData(train_x, train_y, test_x):
    train_x[train_x == 'F'] = '0'
    train_x[train_x == 'I'] = '1'
    train_x[train_x == 'M'] = '2'
    train_x = train_x.astype(np.float)

    test_x[test_x == 'F'] = '0'
    test_x[test_x == 'I'] = '1'
    test_x[test_x == 'M'] = '2'
    test_x = test_x.astype(np.float)

    train_x = np.append(train_x, np.ones((len(train_x), 1)), axis=1)
    test_x = np.append(test_x, np.ones((len(test_x), 1)), axis=1)

    train_y = train_y.astype(np.int)

    return train_x, train_y, test_x


def unisonShuffledCopies(firstArray, secondArray):
    assert len(firstArray) == len(secondArray)
    p = np.random.permutation(len(firstArray))
    return firstArray[p], secondArray[p]


def runTestSet(test_x, perceptronWeights, svmWeights, passiveAggressiveWeights):
    for xi in test_x:
        perceptron_yhat = np.argmax(np.dot(perceptronWeights, xi))
        svm_yhat = np.argmax(np.dot(svmWeights, xi))
        pa_yhat = np.argmax(np.dot(passiveAggressiveWeights, xi))
        print(f"perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}")


if __name__ == "__main__":
    # Data preparation.
    train_x_filename, train_y_filename, test_x_filename = sys.argv[1], sys.argv[2], sys.argv[3]

    train_x = np.genfromtxt(train_x_filename, dtype="str", delimiter=",")
    train_y = np.genfromtxt(train_y_filename, dtype="str", delimiter=",")
    test_x = np.genfromtxt(test_x_filename, dtype="str", delimiter=",")
    train_x, train_y, test_x = prepareData(train_x, train_y, test_x)

    # Shuffle data.
    train_x, train_y = unisonShuffledCopies(train_x, train_y)

    perceptron = Perceptron(learningRate=0.0001, iterationsAmount=100)
    perceptronWeights = perceptron.fit(train_x, train_y)

    svm = SupportVectorMachine(learningRate=0.001, iterationsAmount=100, lamda=0.1)
    svmWeights = svm.fit(train_x, train_y)

    passiveAggressive = PassiveAggressive(iterationsAmount=20)
    passiveAggressiveWeights = passiveAggressive.fit(train_x, train_y)

    runTestSet(test_x, perceptronWeights, svmWeights, passiveAggressiveWeights)
