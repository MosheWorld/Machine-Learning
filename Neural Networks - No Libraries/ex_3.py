import sys
import random
import numpy as np


class ANN(object):
    def __init__(self, layers_architecture):
        self.layers = layers_architecture
        self.layers_size = len(layers_architecture)
        self.biases = [np.random.uniform(-0.8, 0.8, (y, 1)) for y in self.layers[1:]]
        self.weights = [np.random.uniform(-0.8, 0.8, (y, x)) for x, y in zip(self.layers[:-1], self.layers[1:])]

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        expo = np.exp(z - np.max(z))
        return expo / expo.sum()

    def fit(self, train_data, epochs, eta, batch_size, validation_data=None):
        train_data_size = len(train_data)

        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[s:s+batch_size] for s in range(0, train_data_size, batch_size)]

            for batch in batches:
                biases_derivative = [np.zeros(b.shape) for b in self.biases]
                weights_derivative = [np.zeros(w.shape) for w in self.weights]

                for x, y in batch:
                    weights_change_rate, biases_change_rate = self.backward_propagation(x, y)
                    biases_derivative = [bd + bcr for bd, bcr in zip(biases_derivative, biases_change_rate)]
                    weights_derivative = [wd + wcr for wd, wcr in zip(weights_derivative, weights_change_rate)]

                self.weights = [w-(eta/len(batch))*wd for w, wd in zip(self.weights, weights_derivative)]
                self.biases = [b-(eta/len(batch))*bd for b, bd in zip(self.biases, biases_derivative)]
                
            if validation_data:
                ratio, percentage = self.evaluate(validation_data)
                print("Epoch {}: Ratio, {},  Percentage, {}".format(i, ratio, percentage))

    def forward_propagation(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.relu(np.dot(w, x) + b)
        return x

    def forward_propagation_full(self, x):
        current_layer = x
        activation_layers = [x] 
        z_list = []

        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], current_layer) + self.biases[i]
            z_list.append(z)
            current_layer = self.relu(z)
            activation_layers.append(current_layer)
        
        z = np.dot(self.weights[-1], current_layer) + self.biases[-1]
        z_list.append(z)
        current_layer = self.softmax(z) 
        activation_layers.append(current_layer)

        return activation_layers, z_list

    def backward_propagation(self, x, y):
        biases_change_rate = [np.zeros(b.shape) for b in self.biases]
        weights_change_rate = [np.zeros(w.shape) for w in self.weights]

        # Forward Propagation.
        activation_layers, z_list = self.forward_propagation_full(x)

        # Backward Propagation.
        change_rate = (activation_layers[-1] - y)
        biases_change_rate[-1] = change_rate
        weights_change_rate[-1] = np.dot(change_rate, activation_layers[-2].transpose())

        for l in range(2, self.layers_size):
            z = z_list[-l]
            change_rate = np.dot(self.weights[-l+1].transpose(), change_rate) * self.relu_prime(z)
            biases_change_rate[-l] = change_rate
            weights_change_rate[-l] = np.dot(change_rate, activation_layers[-l-1].transpose())

        return weights_change_rate, biases_change_rate 

    def evaluate(self, validation_data):
        size = len(validation_data)
        counter = 0

        for x, y in validation_data:
            argmax_feedforward = np.argmax(self.forward_propagation(x))
            argmax_real = np.argmax(y)
            if argmax_feedforward == argmax_real:
                counter += 1

        return str(counter) + "/" + str(size), str((counter/size) * 100)

    def predict_test_set(self, test_x):
        prediction_list = []

        for x in test_x:
            prediction_list.append(np.argmax(self.forward_propagation(x)))

        return prediction_list


def number_to_vector(index):
    vector = np.zeros((10, 1))
    vector[int(index)] = 1.0
    return vector


def load_data(train_x_filename, train_y_filename, test_x_filename):
    train_x = np.loadtxt(train_x_filename) / 255
    train_y = np.loadtxt(train_y_filename)
    test_x = np.loadtxt(test_x_filename) / 255

    train_inputs = [np.reshape(x, (784, 1)) for x in train_x]
    train_outputs = [number_to_vector(y) for y in train_y]
    test_inputs = [np.reshape(x, (784, 1)) for x in test_x]

    return list(zip(train_inputs, train_outputs)), test_inputs


def load_data_with_validation(train_x_filename, train_y_filename, test_x_filename):
    train_x = np.loadtxt(train_x_filename) / 255
    train_y = np.loadtxt(train_y_filename)
    test_x = np.loadtxt(test_x_filename) / 255
    split_size = int(len(train_x) * 0.8)

    validation_x = train_x[split_size:]
    validation_y = train_y[split_size:]
    train_x = train_x[:split_size]
    train_y = train_y[:split_size]

    train_inputs = [np.reshape(x, (784, 1)) for x in train_x]
    train_outputs = [number_to_vector(y) for y in train_y]
    test_inputs = [np.reshape(x, (784, 1)) for x in test_x]
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_x]
    validation_outputs = [number_to_vector(y) for y in validation_y]

    train_data = list(zip(train_inputs, train_outputs))
    validation_data = list(zip(validation_inputs, validation_outputs))

    return train_data, validation_data, test_inputs


def output_to_file_system(output_filename, output):
    file = open(output_filename, "w")
    file.write(output)
    file.close()


if __name__ == "__main__":
    train_x_filename, train_y_filename, test_x_filename = sys.argv[1], sys.argv[2], sys.argv[3]
    # train_x_filename, train_y_filename, test_x_filename = "train_x", "train_y", "test_x"
    train_data, test_inputs = load_data(train_x_filename, train_y_filename, test_x_filename)
    # train_data, validation_data, test_inputs = load_data_with_validation(train_x_filename, train_y_filename, test_x_filename)

    ann = ANN([784, 128, 128, 10])
    ann.fit(train_data=train_data, epochs=40, eta=0.001, batch_size=1)
    prediction_list = ann.predict_test_set(test_inputs)
    output = "\n".join(map(str, prediction_list))
    output_to_file_system(output_filename="test_y", output=output)
