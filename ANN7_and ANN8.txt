import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

on
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


np.random.seed(42)

weights_0 = 2 * np.random.random((2, 4)) - 1
weights_1 = 2 * np.random.random((4, 1)) - 1

for i in range(10000):
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, weights_0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1))
    
    error = y - layer_2
    
    delta_2 = error * sigmoid_derivative(layer_2)
    delta_1 = delta_2.dot(weights_1.T) * sigmoid_derivative(layer_1)
    

    weights_1 += layer_1.T.dot(delta_2)
    weights_0 += layer_0.T.dot(delta_1)

output = sigmoid(np.dot(sigmoid(np.dot(X, weights_0)), weights_1))
print("Predicted Output:")
print(output)
