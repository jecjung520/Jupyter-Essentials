import numpy as np
import matplotlib.pyplot as plt

def GaussianDiscriminantAnalysis(X1, X2):
    n1 = X1.shape[1]
    n2 = X2.shape[1]
    
    # prior probabilities of class 1 and class 2
    prior1 = n1 / (n1 + n2)
    prior2 = n2 / (n1 + n2)
    
    # mean vectors of two classes
    mean1 = np.mean(X1, axis=1).reshape(-1,1)
    mean2 = np.mean(X2, axis=1).reshape(-1,1)
    
    # covariance matrices of two classes
    cov1 = np.zeros((X1.shape[0], X1.shape[0]))
    for i in range(n1):
        x = X1[:, i].reshape(-1, 1)
        cov1 += (x - mean1).dot((x - mean1).T)
    cov1 /= n1
    
    cov2 = np.zeros((X2.shape[0], X2.shape[0]))
    for i in range(n2):
        x = X2[:, i].reshape(-1, 1)
        cov2 += (x - mean2).dot((x - mean2).T)
    cov2 /= n2
    
    # mean covariance matrix
    cov = (n1 * cov1 + n2 * cov2) / (n1 + n2)
    
    # w in the decision boundary
    w = np.linalg.inv(cov).dot(mean1 - mean2)
    
    # w0 in the decision boundary
    w0 = -0.5 * mean1.T.dot(np.linalg.inv(cov)).dot(mean1) + 0.5 * mean2.T.dot(np.linalg.inv(cov)).dot(mean2) + np.log(prior1/prior2)
    
    return prior1, mean1, mean2, cov1, cov2, cov, w, w0
    
def plot_GDA(X1, X2, w, w0):
    # Plot the data points
    plt.scatter(X1[0, :], X1[1, :], marker='o', color='r', label='Class 1')
    plt.scatter(X2[0, :], X2[1, :], marker='x', color='b', label='Class 2')
    plt.legend()
    
    # Plot the decision boundary
    x_min = np.min([X1[0, :], X2[0, :]]) - 1
    x_max = np.max([X1[0, :], X2[0, :]]) + 1
    y_min = np.min([X1[1, :], X2[1, :]]) - 1
    y_max = np.max([X1[1, :], X2[1, :]]) + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = w0 + w[0] * xx + w[1] * yy
    plt.contour(xx, yy, Z, levels=[0], colors='k')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gaussian Discriminant Analysis')
    plt.show()