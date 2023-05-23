import numpy as np
X = np.array([[1, 1, 1, -1], [-1, -1, 1, 1]])
Y = np.array([[1, -1], [-1, 1]])
W = np.dot(Y.T, X)

def bam(x):
    y = np.sign(np.dot(W, x))
    return y

x_test = np.array([1, -1, -1, -1])
y_test = bam(x_test)


print("Input x:", x_test)
print("Output y:", y_test)
