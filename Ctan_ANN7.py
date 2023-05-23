import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class BackpropagationNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(2, 2)
        self.biases1 = np.random.rand(2)
        self.weights2 = np.random.rand(2, 1)
        self.biases2 = np.random.rand(1)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # forward propagation
            hidden_layer_activation = sigmoid(np.dot(X, self.weights1) + self.biases1)
            output = sigmoid(np.dot(hidden_layer_activation, self.weights2) + self.biases2)

            
            error = y - output

            # backpropagation
            output_delta = error * sigmoid_derivative(output)
            hidden_delta = np.dot(output_delta, self.weights2.T) * sigmoid_derivative(hidden_layer_activation)

        
            self.weights2 += np.dot(hidden_layer_activation.T, output_delta)
            self.biases2 += np.sum(output_delta, axis=0)
            self.weights1 += np.dot(X.T, hidden_delta)
            self.biases1 += np.sum(hidden_delta, axis=0)

    def predict(self, X):
        hidden_layer_activation = sigmoid(np.dot(X, self.weights1) + self.biases1)
        output = sigmoid(np.dot(hidden_layer_activation, self.weights2) + self.biases2)
        return output

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

bpn = BackpropagationNetwork()
bpn.train(X, y, epochs=10000)

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = bpn.predict(test_data)

for x, prediction in zip(test_data, predictions):
    print(f"Input: {x}, Prediction: {prediction}")
