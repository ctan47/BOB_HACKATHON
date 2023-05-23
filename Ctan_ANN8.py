import numpy as np

def initialize_weights(n_features, n_categories):
    return np.random.rand(n_categories, n_features)

def normalize(data):
    return data / np.sum(data)

def update_activation(data, weights, vigilance):
    activations = np.dot(data, weights.T)
    return normalize(np.minimum(activations, vigilance))

def update_weights(data, weights, activation, lr):
    return weights + lr * np.outer(activation, data)

def train_art1(data, n_categories, vigilance, lr, max_epochs):
    n_features = data.shape[1]
    weights = initialize_weights(n_features, n_categories)

    for _ in range(max_epochs):
        for i in range(len(data)):
            x = data[i]
            a = update_activation(x, weights, vigilance)
            weights = update_weights(x, weights, a, lr)

    return weights

def predict(data, weights):
    activations = np.dot(data, weights.T)
    return np.argmax(activations, axis=1)

# Example usage
data = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
labels = np.array([0, 1, 2, 3])

n_categories = 4
vigilance = 0.9
lr = 0.1
max_epochs = 17

trained_weights = train_art1(data, n_categories, vigilance, lr, max_epochs)
predictions = predict(data, trained_weights)

print("Predictions:", predictions)
