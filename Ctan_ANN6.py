import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(2, 1)
        self.bias = np.random.rand(1)

    def train(self, X, y, epochs):
        for i in range(epochs):
            output = self.predict(X)
            error = y - output
            #backward_prop
            delta = error * output * (1 - output)
            self.weights += np.dot(X.T, delta)
            self.bias += np.sum(delta)

    def predict(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias))) #forward_Prop

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

nn = NeuralNetwork()
nn.train(X, y, epochs=1000)

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.predict(test_data)

for x, prediction in zip(test_data, predictions):
    print(f"Input: {x}, Prediction: {prediction}")
