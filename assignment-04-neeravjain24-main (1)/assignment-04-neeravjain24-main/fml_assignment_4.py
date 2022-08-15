# -*- coding: utf-8 -*-
"""fml_assignment_4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D9ZEF7dlKb-dnLgloIWIf7Fowtkj25Rp
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import datasets 

plt.close('all') #close any open plots

def eta_function(u, d, m):
    u = u ** m
    n = np.sum(u * d, axis=1) / np.sum(u, axis=1)
    return n

def clustering_updating_function(x, u, m):
    um = u ** m
    v = um.dot(x.T) / np.atleast_2d(um.sum(axis=1)).T
    return v

def criterion_for_fcm(x, v, n, m, metric):
    d = cdist(x.T, v, metric=metric).T
    d = np.fmax(d, np.finfo(x.dtype).eps)
    exp = -2. / (m - 1)
    d2 = d ** exp
    u = d2 / np.sum(d2, axis=0, keepdims=1)
    return u, d


def criterion_for_pcm(x, v, n, m, metric):
    d = cdist(x.T, v, metric=metric)
    d = np.fmax(d, np.finfo(x.dtype).eps)
    d2 = (d ** 2) / n
    exp = 1. / (m - 1)
    d2 = d2.T ** exp
    u = 1. / (1. + d2)
    return u, d


def cmean_clustering(x, c, m, e, max_iterations, criterion_function, metric="euclidean", v0=None, n=None):
    if not x.any() or len(x) < 1 or len(x[0]) < 1:
        print("Error: Data is in incorrect format")
        return
    S, N = x.shape
    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")
    if not m:
        print("Error: Fuzzifier must be greater than 1")
        return
    if v0 is None:
        xt = x.T
        v0 = xt[np.random.choice(xt.shape[0], c, replace=False), :]
    v = np.empty((max_iterations, c, S))
    v[0] = np.array(v0)
    u = np.zeros((max_iterations, c, N))
    t = 0
    while t < max_iterations - 1:
        u[t], d = criterion_function(x, v[t], n, m, metric)
        v[t + 1] = clustering_updating_function(x, u[t], m)
        if np.linalg.norm(v[t + 1] - v[t]) < e:
            break
        t += 1
    return v[t], v[0], u[t - 1], u[0], d, t

def fucntion_for_fcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):
    return cmean_clustering(x, c, m, e, max_iterations, criterion_for_fcm, metric, v0=v0)


def function_for_pcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):
    v, v0, u, u0, d, t = fucntion_for_fcm(x, c, m, e, max_iterations, metric=metric, v0=v0)
    n = eta_function(u, d, m)
    return cmean_clustering(x, c, m, e, t, criterion_for_pcm, metric, v0=v, n=n)


n_samples = 1500

blobs, y_blobs = datasets.make_blobs(n_samples=n_samples)

transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(blobs, transformation)
y_aniso = y_blobs
 
X_varied, y_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])

X_filtered = np.vstack((X_varied[y_varied == 0][:500], X_varied[y_varied == 1][:100], X_varied[y_varied == 2][:10]))
y_filtered = np.array([0]*500 + [1]*100 + [2]*10)

c = 3
fuzzifier = 1.2
error = 0.001
maxiter = 100

def clustering_verification_function(x, c, v, u, labels):

    ssd_actual = 0

    for i in range(c):
        # All points in class
        x1 = x[labels == i]
        # Mean of class
        m = np.mean(x1, axis=0)

        for pt in x1:
            ssd_actual += np.linalg.norm(pt - m)

    clm = np.argmax(u, axis=0)
    ssd_clusters = 0

    for i in range(c):
        # Points clustered in a class
        x2 = x[clm == i]

        for pt in x2:
            ssd_clusters += np.linalg.norm(pt - v[i])

    print(ssd_clusters / ssd_actual)

from sklearn.decomposition import PCA


def plot(x, v, u, c, labels=None):

    ax = plt.subplots()[1]

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)

    x = PCA(n_components=2).fit_transform(x).T

    for j in range(c):
        ax.scatter(
            x[0][cluster_membership == j],
            x[1][cluster_membership == j],
            alpha=0.5,
            edgecolors="none")

    ax.legend()
    ax.grid(True)
    plt.show()


v, v0, u, u0, d, t = function_for_pcm(blobs.T, c, fuzzifier, error, maxiter)
plt.figure(figsize=(12, 12))
plt.subplot(431)
plt.scatter(blobs[:, 0], blobs[:, 1], c=u[0,:])
plt.title("Blobs")
plt.subplot(432)
plt.scatter(blobs[:, 0], blobs[:, 1], c=u[1,:])
plt.title("Blobs")
plt.subplot(433)
plt.scatter(blobs[:, 0], blobs[:, 1], c=u[2,:])
plt.title("Blobs")


v, v0, u, u0, d, t = function_for_pcm(X_aniso.T, c, fuzzifier, error, maxiter)
plt.subplot(434)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=u[0,:])
plt.subplot(435)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=u[1,:])
plt.subplot(436)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=u[2,:])
plt.title("Anisotropicly Distributed Blobs")


v, v0, u, u0, d, t = function_for_pcm(X_varied.T, c, fuzzifier, error, maxiter)
plt.subplot(437)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=u[0,:])
plt.subplot(438)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=u[1,:])
plt.subplot(439)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=u[2,:])
plt.title("Unequal Variance - Only Blobs")

v, v0, u, u0, d, t = function_for_pcm(X_filtered.T, c, fuzzifier, error, maxiter)
plt.subplot(4,3,10)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=u[0,:])
plt.subplot(4,3,11)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=u[1,:])
plt.subplot(4,3,12)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=u[2,:])
plt.title("Unequal Variance - Only Blobs")

plt.savefig('.jpg')
