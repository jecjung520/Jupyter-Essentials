import numpy as np
import matplotlib.pyplot as plt

def LDA(X1, X2):
    n1 = X1.shape[1]
    n2 = X2.shape[1]
    
    # calculate means of class 1 and class 2
    mean1 = np.mean(X1, axis=1).reshape(-1, 1)
    mean2 = np.mean(X2, axis=1).reshape(-1, 1)
    
    # calculate scatter matrices for class 1 and class 2
    S1 = np.zeros((X1.shape[0], X1.shape[0]))
    for i in range(X1.shape[1]):
        x = X1[:, i].reshape(-1, 1)
        S1 += (x - mean1).dot((x - mean1).T)
    S2 = np.zeros((X2.shape[0], X2.shape[0]))
    for i in range(X2.shape[1]):
        x = X2[:, i].reshape(-1, 1)
        S2 += (x - mean2).dot((x - mean2).T)
    
    # calculate the within-class scatter matrix Sw
    Sw = S1 + S2
    
    # calculate the between-class scatter matrix Sb
    Sb = np.dot((mean1 - mean2), (mean1 - mean2).T)
    Swinv = np.linalg.inv(Sw);
    
    # calculate the optimum projection vector w
    w, v = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    w = v[:, np.argmax(w)]
    w = np.absolute(w)
    
    # calculate the projected space of Class1 and Class2
    Y1 = w.T.dot(X1)
    Y2 = w.T.dot(X2)
    
    # calculate mean and covariance on the y-space
    mu_Y1 = np.mean(Y1.reshape(-1, 1))
    mu_Y2 = np.mean(Y2.reshape(-1, 1))

    Var_Y1 = np.cov(Y1)
    Var_Y2 = np.cov(Y2)
    
    p1 = n1 / (n1 + n2)
    
    # calculate classification threshold a, b, c
    a = Var_Y1 - Var_Y2
    b = 2*mu_Y1*Var_Y2 - 2*mu_Y2*Var_Y1;
    c = mu_Y2**2*Var_Y1-mu_Y1**2*Var_Y2+Var_Y1*Var_Y2*np.log((p1**2*Var_Y2)/((1-p1)**2*Var_Y1))
    
    # calculate decision threshold T
    s1 = (-b + np.sqrt(np.absolute(b**2-4*a*c)))/(2*a)
    s2 = (-b - np.sqrt(np.absolute(b**2-4*a*c)))/(2*a)
    
    return mean1, mean2, S1, S2, Sw, Swinv, w, Y1, Y2, mu_Y1, mu_Y2, Var_Y1, Var_Y2, a, b, c, s1, s2

def plot_LDA(X1, X2, w):
    fig, ax = plt.subplots()
    
    # plot training data points
    ax.scatter(X1[0,:], X1[1,:], c='r', marker='o', label='Class 1')
    ax.scatter(X2[0,:], X2[1,:], c='b', marker='o', label='Class 2')
    
    # plot the decision boundary
    ax.quiver(0, 0, w[0], X2, color='g', angles='xy', scale_units='xy', scale=1, label='Projection Vector')
    
    ax.legend()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def plot_pdfs(mean1, mean2, std1, std2):
    y = np.linspace(-15, 30, 1000)
    y1 = np.exp(-(y - mean1)**2 / (2 * std1**2)) / (np.sqrt(2 * np.pi) * std1)
    y2 = np.exp(-(y - mean2)**2 / (2 * std2**2)) / (np.sqrt(2 * np.pi) * std2)
    plt.plot(y, y1, 'r', label='Class 1')
    plt.plot(y, y2, 'b', label='Class 2')
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('f(y|Ck)')
    plt.show()
    
def plot_posteriors(mean1, mean2, std1, std2):
    y = np.linspace(10, 25, 1000)
    y1 = np.exp(-(y - mean1)**2 / (2 * std1**2)) / (np.sqrt(2 * np.pi) * std1)
    y2 = np.exp(-(y - mean2)**2 / (2 * np.pi) * std2**2) / (np.sqrt(2 * np.pi) * std2)
    f_C1_y = y1 * 0.5 / (y1 * 0.5 + y2 * 0.5)
    f_C2_y = y2 * 0.5 / (y1 * 0.5 + y2 * 0.5)
    plt.plot(y, f_C1_y, 'r', label='f(C1|y)')
    plt.plot(y, f_C2_y, 'b', label='f(C2|y)')
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('f(Ck|y)')
    plt.show()