# Linear Regression Gradient Descent with Single Feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

house_sales = pd.read_csv('house_sales.csv')

sqft_living = house_sales[['sqft_living']].values.flatten()[:, np.newaxis]
sqft_lot = house_sales[['sqft_lot']].values.flatten()[:, np.newaxis]
bedrooms = house_sales[['bedrooms']].values.flatten()[:, np.newaxis]
bathrooms = house_sales[['bathrooms']].values.flatten()[:, np.newaxis]

data_m = np.hstack((sqft_living, sqft_lot, bedrooms, bathrooms))
stdm = np.std(data_m, axis=0)
mum = np.mean(data_m, axis=0)
data_m = (data_m - mum) / stdm
ones = np.ones((len(data_m), 1))
data_m = np.hstack((ones, data_m))
print('data_m', data_m)

# setup features
data_x = house_sales[['sqft_living']].values.flatten()[:, np.newaxis]
stdx = np.std(data_x, axis=0)
mux = np.std(data_x, axis=0)
data_x = (data_x - mux) / stdx
data_x = np.hstack((np.ones_like(data_x), data_x))

print('data_x', data_x)

# setup targets
data_y = house_sales[['price']].values.flatten()[:, np.newaxis]

# separate into training and testin datasets
order = np.random.permutation(len(data_x))
portion = 10000
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]


def get_gradient(w, x, y):
    # matrix multiply x by weights
    y_estimate = x.dot(w).flatten()

    # calculate error between actual values and estimates
    error = (y.flatten() - y_estimate)

    mse = (1 / len(x)) * np.sum(np.power(error, 2))

    # calculate the gradient
    # error.dot(x) multiplies the error for each observation
    # by the value of the feature use to predict
    gradient = -(1 / len(x)) * error.dot(x)

    return gradient, mse


# variables for use in gradient descent algorithm
w = [0, 0]
alpha = 0.5
tolerance = 1e-5

# Perform Gradient Descent
iterations = 1
while True:
    gradient, error = get_gradient(w, train_x, train_y)
    new_w = w - alpha * gradient

    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print('converged')
        break

    # Print error every 50 iterations
    if iterations % 100 == 0:
        print("Iteration:", iterations)
        print("Error:", np.sum(error))

    iterations += 1
    w = new_w

print('w =', w)
print('Test Cost = ', get_gradient(w, test_x, test_y)[1])
print('xdotw', data_x.dot(w))
print('x', data_x[:, 1])

plt.plot(data_x[:,1], data_x.dot(w), c='g', label='Model')
plt.scatter(train_x[:,1], train_y, c='b', label='Train Set')
plt.scatter(test_x[:,1], test_y, c='r', label='Test Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
