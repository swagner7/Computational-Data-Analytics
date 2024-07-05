import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy
import random

# run PCA --------------------------
def run_PCA(data):
    import math

    rows = data.shape[0]  # initialize rows constant for looping
    cols = data.shape[1]  # initialize columns constant for looping
    data = data.T # transpose data
    avg = np.mean(data, axis = 1, keepdims = True) # calculate the mean of the data
    data = data - avg # subtract the mean from the samples
    C = np.dot(data, data.T)/rows # calculate weighting vectors, C
    S, W = scipy.sparse.linalg.eigs(C, k = 2) # perform eigen decomp
    S = S.real # remove imaginary elements
    W = W.real
    PC_1 = np.dot(W[:, 0].T, data) / math.sqrt(S[0])  # 1st principal component
    PC_2 = np.dot(W[:, 1].T, data) / math.sqrt(S[1])  # 2nd principal component

    return PC_1, PC_2

# load data
data = scipy.io.loadmat('isomap.mat')['images'].T

# compute weighted adjacency matrix, A
A = pairwise_distances(data, metric = 'l1')
nodes = A.shape[0]  # number of nodes
neighbors = 101
for i in range(nodes):
    # find the threshold epsilon for each node so that it has at least 100 neighbors
    epsilon = np.partition(A[i], neighbors)[neighbors - 1]
    for j in range(nodes):
        if A[i, j] > epsilon:
            A[i, j] = 1e6  # set distance to be large if two nodes are unconnected

A = (A + A.T) / 2  # make the similarity matrix symmetric

# visualize adjacency matrix via an image
fig, ax = plt.subplots(1,1)
image = ax.imshow(A)
colorbar = plt.colorbar(image)
ax.set_title('ISOMAP Adjacency Matrix')
# plt.show()

# find shortest path using provided scipy algorithm
D = scipy.sparse.csgraph.shortest_path(A)

# calculate C
H = np.eye(nodes) - np.ones((nodes, nodes)) / nodes # calculate centering matrix, H
C = H @ (D*D)
C = (C @ H) / -2
C = (C + C.T) / 2

# eigen decomposition of C
eigenvalues, eigenvectors = np.linalg.eig(C)

# projection
Z = eigenvectors[:, 0:2] @ np.diag(np.sqrt(eigenvalues[:2]))

# selected images to show in the 2D scatter plot
# show_images = [100, 445, 234, 1, 60, 263, 189, 320, 559, 648, 390]
show_images = random.sample(range(0, 648), 20)


fig, ax = plt.subplots()
ax.plot(Z[:, 0], Z[:, 1], '.k')
ax.set_title('ISOMAP')
for i in show_images:
    img = data[i].reshape(64, 64).T

    img_box = OffsetImage(img, zoom = 0.4)
    ab = AnnotationBbox(img_box, -Z[i])
    ax.add_artist(ab)


# run PCA (from Problem #2)
z = np.zeros((nodes, 2))
z[:, 0], z[:, 1] = run_PCA(data)

fig, ax = plt.subplots()
ax.plot(z[:, 0], z[:, 1], '.k')
ax.set_title('PCA')
for i in show_images:
    img = data[i].reshape(64, 64).T

    img_box = OffsetImage(img, zoom = 0.4)
    ab = AnnotationBbox(img_box, -z[i])
    ax.add_artist(ab)

plt.show()