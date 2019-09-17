# Linear Regression Gradient Descent with Multiple Features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

house_sales = pd.read_csv('house_sales.csv')

features = house_sales.iloc[:, 3:9].values

data_m = features
stdm = np.std(data_m, axis=0)
mum = np.mean(data_m, axis=0)
data_m = (data_m - mum) / stdm
ones = np.ones((len(data_m), 1))
data_m = np.hstack((ones, data_m))

print(np.shape(data_m))

# setup targets
data_y = house_sales[['price']].values.flatten()[:, np.newaxis]

# separate into training and testing datasets
order = np.random.permutation(len(data_m))
portion = 10000
test_mx = data_m[order[:portion]]
test_my = data_y[order[:portion]]
train_mx = data_m[order[portion:]]
train_my = data_y[order[portion:]]

def get_gradient_multi(w, x, y):
    y_estimate = x.dot(w).flatten()[:, np.newaxis]
    error = (y - y_estimate)

    mse = (1 / len(x)) * np.sum(np.power(error, 2))

    gradient = -(1 / len(x)) * np.transpose(x).dot(error)

    return gradient, mse


# variables for use in gradient descent algorithm
w = np.zeros((np.size(data_m, axis=1), 1))
alpha = 0.05
tolerance = 1e-5

# Perform Gradient Descent
iterations = 1
while True:
    gradient, error = get_gradient_multi(w, train_mx, train_my)
    new_w = w - alpha * gradient

    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print('converged')
        break

    # Print error every 10 iterations
    if iterations % 10 == 0:
        print("Iteration:", iterations)
        print("Error:", np.sum(error))

    iterations += 1
    w = new_w

print('w =', w)
print('Test Cost = ', get_gradient_multi(w, test_mx, test_my)[1])

# sort model output for plot
datadotw = data_m.dot(w).flatten()
data = data_m[:, 2]
sorted_index = np.argsort(data)
sorted_x = data[sorted_index]
sorted_y = datadotw[sorted_index]

print('features', test_mx[1, :])
print('actual price', test_my[1, :])
print('predicted price', test_mx[1, :].dot(w))


# plot
plt.plot(sorted_x, sorted_y, c='g', label='Model')
plt.scatter(train_mx[:,1], train_my, c='b', label='Train Set')
plt.scatter(test_mx[:,1], test_my, c='r', label='Test Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()
