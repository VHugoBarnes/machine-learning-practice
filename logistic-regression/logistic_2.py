# How to calculate the cross entropy error
import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] - 2 * np.ones((50,D))

T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X), axis=1)

# Randomly initialize the weights
w = np.random.randn(D+1)

# Calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    E = 0

    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    
    return E

print(cross_entropy(T, Y))

w = np.array([0, 4, 4])
z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy(T, Y))
