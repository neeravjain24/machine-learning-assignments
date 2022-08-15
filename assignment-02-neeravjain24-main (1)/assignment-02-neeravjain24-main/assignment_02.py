import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.close('all')  # close any open plots

# data_uniform
data_uniform = np.load('data_set.npz')
x_train = data_uniform['arr_0'][:, 0]
y_train = data_uniform['arr_0'][:, 1]
x_val = data_uniform['arr_1'][:, 0]
y_val = data_uniform['arr_1'][:, 1]
x_test = data_uniform['arr_2'][:, 0]
y_test = data_uniform['arr_2'][:, 1]

# M=N
M = len(x_train)

# setting s=0.01
s = 0.01

# function to calculate target value using known true function y=x/(x+1)


def true_val_func(x):
    y = 3*(x+np.sin(x))*np.exp(-(x**2))

    return y

# function to calculate weights


def fit_data_rbf(x, t, m, sd, eta, epochs, alpha, l1):
    # initialize l2
    l2 = 1 - l1

    # initialize random weights
    w = np.random.random((len(x),))

    # calculating feature matrix
    X = np.array([np.exp(-(x-mu)**2/(2*sd**2)) for mu in m]).T
    t.reshape(t.shape[0], 1)

    # adding a diagonal matrix of 0.001 to remove Simgular matrix error
    np.fill_diagonal(X, X.diagonal() + 0.001)

    # grdient desent
    while epochs != 0:
        temp = X.T@(X@w - t) + (alpha*(l1*np.sign(w) + 2*l2*w))
        w = w - eta*temp
        epochs -= 1

    return w

# function to apply model with evenly spaced mu


def rbf_model_even(x, t, m, sd, eta, epochs, alpha, l1):
    mu_list = []
    x1_split = np.array_split(x, m)
    for i in x:
        mu = np.median(i)
        mu_list.append(mu)

    weights = fit_data_rbf(x, t, mu_list, sd, eta, epochs, alpha, l1)

    return mu_list, weights


def predict_by_rbf_model(x, m, w, sd):
    # converting into feature matrix
    X = np.array([np.exp(-(x-mu)**2/(2*sd**2)) for mu in m]).T

    # prediction of Y values using pre-trained weights
    Y = X@w

    return Y


def mean_abs_err(y, t):
    errs = []
    for i in range(0, len(y)):
        errs.append(abs(t[i]-y[i]))
    errs = np.array(errs)
    return np.mean(errs)


def get_plots(x1, t1, x2, t2, x3, t3, M, s, eta, ep, alpha, l1):
    fig, axs = plt.subplots(1, figsize=(15, 15))
    fig.suptitle('Plots with M='+str(M)+' and s='+str(s), fontsize=16)
    fig.tight_layout(pad=5.0)
    plt.subplots_adjust(bottom=0.3, right=0.8, top=0.925)

    # plot 1
    axs.scatter(x1, t1)
    axs.set_title('Training data points')

    # plot 2
    mu_mat, weights = rbf_model_even(x1, t1, M, s, eta, ep, alpha, l1)
    X = x2
    Y = predict_by_rbf_model(X, mu_mat, weights, s).T
    axs.plot(X, Y)

    # plot 4
    X = x2
    Y = true_val_func(X).T
    axs.plot(X, Y)

    axs.set(xlabel='x', ylabel='y or t')
    axs.legend(['Predicted test values', 'True test values', 'Training data'])
    axs.set_title('Prediction on Test Dataset')


def get_3d_plot(x1, t1, x2, t2, x3, t3, M, s, eta, ep):
    alpha_range = np.arange(0, 10, 1)
    lamda1_range = np.arange(0, 1, 0.01)

    # plot 1
    mean_errs_in_train = np.zeros((10, 100))
    mean_errs_in_val = np.zeros((10, 100))
    mean_errs_in_test = np.zeros((10, 100))
    for a in alpha_range:
        k = 0
        for l1 in lamda1_range:
            mu_mat, weights = rbf_model_even(x1, t1, M, s, eta, ep, a, l1)
            y1 = predict_by_rbf_model(x1, mu_mat, weights, s)
            y2 = predict_by_rbf_model(x2, mu_mat, weights, s)
            y3 = predict_by_rbf_model(x3, mu_mat, weights, s)

            mean_errs_in_train[a][k] = mean_abs_err(y1, t1)
            mean_errs_in_val[a][k] = mean_abs_err(y2, t2)
            mean_errs_in_test[a][k] = mean_abs_err(y3, t3)
            k += 1

    X, Y = np.meshgrid(alpha_range, lamda1_range)
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(X, Y, mean_errs_in_train.T)
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Lamda 1')
    ax1.set_zlabel('Mean Absolute Errors')
    plt.savefig('1')

    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(X, Y, mean_errs_in_val.T)
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Lamda 1')
    ax2.set_zlabel('Mean Absolute Errors')
    plt.savefig('2')

    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X, Y, mean_errs_in_test.T)
    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('Lamda 1')
    ax3.set_zlabel('Mean Absolute Errors')
    plt.savefig('3')

    alpha_range = np.arange(0, 10, 1)
    lamda1_range = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(alpha_range, lamda1_range)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, mean_errs_in_train.T)
    ax.plot_surface(X, Y, mean_errs_in_val.T)
    ax.plot_surface(X, Y, mean_errs_in_test.T)
    plt.savefig('5')


get_plots(x_train, y_train, x_val, y_val,
          x_test, y_test, M, s, 0.1, 500, 1, 0.4)
get_3d_plot(x_train, y_train, x_val, y_val, x_test, y_test, M, s, 0.01, 200)
