import matplotlib.pyplot as plt
import numpy as np

def plot_GDA(X1, X2, w, w0):
    import matplotlib.pyplot as plt

    # plot the training data points
    plt.scatter(X1[0, :], X1[1, :], c='b', label='Class 1')
    plt.scatter(X2[0, :], X2[1, :], c='r', label='Class 2')

    # plot the decision boundary
    x1 = np.array([np.min(X1), np.max(X1)])
    x2 = -(w[0] * x1 + w0) / w[1]
    plt.plot(x1, x2.reshape(-1,), 'g', label='Decision Boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    return plt

plt_obj = plot_GDA(X1, X2, w, w0)
plt_obj.show()
