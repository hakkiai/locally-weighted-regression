import numpy as np
import matplotlib.pyplot as plt

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

def locally_weighted_regression(x, X, Y, tau):
    m = len(X)
    y_pred = np.zeros_like(x)

    for i in range(len(x)):
        weights = np.exp(-(X - x[i])**2 / (2 * tau**2))
        W = np.diag(weights)
        X_2D = X.reshape(-1, 1)
        Y_2D = Y.reshape(-1, 1)
        theta = np.linalg.inv(X_2D.T @ W @ X_2D) @ X_2D.T @ W @ Y_2D
        y_pred[i] = x[i] * theta[0]

    return y_pred

tau = 0.5
x_pred = np.linspace(0, 6, 100)
y_pred = locally_weighted_regression(x_pred, X, Y, tau)

plt.scatter(X, Y, label="Data Points")
plt.plot(x_pred, y_pred, color='red', label="LWR Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()
