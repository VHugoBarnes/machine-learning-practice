# How to calculate the cross entropy error
import numpy as np
import matplotlib.pyplot as plt

# Number of samples we collect
N = 100
# Number of dimensions or features per sample
D = 2

# Matrix of data
# Each row is a sample
# Each column is the value of one feature in each sample
# Here we create a matrix of random numbers from the standard normal distribution
X = np.random.randn(N, D)

# Center the first fifty points at (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
# Center the last fifty points at (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50,D))

# T: Target
# Labels: first fifty are zero, last fifty are one
T = np.array([0]*50 + [1]*50)

# Add a column of ones
# ones = np.array([[1]*N]).T :: old
# In this case, create a matrix of 100x1 filled with 1's
ones = np.ones((N, 1))
# Join a sequence of arrays along an existing axis
# In this case, with axis=1, the X matrix is going to concatenate
# at the end of the ones matrix giving a result like this:
# [[1,1,1, 2.42]]
Xb = np.concatenate((ones,X), axis=1)

# w: weights
# Randomly initialize the weights
w = np.random.randn(D+1)

# Calculate the model output
# by calculating the dot product of Xb and w
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

# Calculate the cross entropy error
def cross_entropy(T, Y):
    E = 0

    # Making the summatory of the cross entropy function
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    
    return E

print(cross_entropy(T, Y))

# try it with our closed-form solution
w = np.array([0, 4, 4])
# Calculate the model output
z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy(T, Y))

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
