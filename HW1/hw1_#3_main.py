import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import show_image
from scipy.sparse import csc_matrix, find

def show_image_function(centroids, H, W):
    N = int((centroids.shape[1]) / (H * W))
    assert (N == 3 or N == 1)

    # Organize the images into rows x cols.
    K = centroids.shape[0]
    COLS = round(math.sqrt(K))
    ROWS = math.ceil(K / COLS)

    COUNT = COLS * ROWS

    plt.clf()
    # plt.hold(True)
    # Set up background at value 100 [pixel values 0-255].
    image = np.ones((ROWS * (H + 1), COLS * (W + 1), N)) * 100
    for i in range(0, centroids.shape[0]):
        r = math.floor(i / COLS)
        c = np.mod(i, COLS)

        image[(r * (H + 1) + 1):((r + 1) * (H + 1)), \
        (c * (W + 1) + 1):((c + 1) * (W + 1)), :] = \
            centroids[i, :].reshape((H, W, N))

    plt.imshow(image.squeeze(), plt.cm.gray)

matFile = sio.loadmat('mnist_10digits.mat')
print(matFile)
# data is 256 x 1100 x 10, consisting of
# 1100 16x16 images of 10 digits.
data = matFile['xtrain']
print(data)
print(data.shape)
pixelno = data.shape[0]
digitno = data.shape[1]
# classno = data.shape[2]

H = 16
W = 16
plt.figure(0)




# Display all images of digits 1 and 0.
# digits_01 = np.concatenate(
#     (np.array(data[:, :, 0]), np.array(data[:, :, 9])), axis=1).T
show_image_function(data, H, W)
plt.title('digit 1 and 0')
plt.show()
os.system(KeyboardInterrupt)
# Create data consisting only 1 and 0.
# x is the images, y is the labels.
x0 = np.array(data[:, :, [0, 9]]).reshape((pixelno, digitno * 2))
x = np.array((data[:, :, [0, 9]]).reshape(
    (pixelno, digitno * 2)), dtype=float)
y = np.concatenate((np.ones((1, digitno)), 2 * np.ones((1, digitno))), axis=1)

# number of data points to work with;
m = x.shape[1]

###############################################################################
# k-means algorithm;
# Greedy algorithm trying to minimize the objective function;
# A highly vectorized version of kmeans.
# Try to run this script several times, and compare the
# (randomized) results.

# run kmeans;
# Number of clusters.
cno = 4

# Randomly initialize centroids with data points;
c = x[:, np.random.randint(x.shape[1], size=(1, cno))[0]]

iterno = 200
for iter in range(0, iterno):
    print("--iteration %d \n" % iter)

    # norm squared of the centroids;
    c2 = np.sum(np.power(c, 2), axis=0, keepdims=True)

    # For each data point x, computer min_j  -2 * x' * c_j + c_j^2;
    # Note that here is implemented as max, so the difference is negated.
    tmpdiff = (2 * np.dot(x.T, c) - c2)
    labels = np.argmax(tmpdiff, axis=1)

    # Update data assignment matrix;
    # The assignment matrix is a sparse matrix,
    # with size m x cno. Only one 1 per row.
    P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, cno))
    count = P.sum(axis=0)

    # Recompute centroids;
    # x*P implements summation of data points assigned to a given cluster.
    c = np.array((P.T.dot(x.T)).T / count)

# Visualize results.
for i in range(0, cno):
    plt.figure(i + 1)
    # Final cluster assignments in P.
    show_image.show_image_function(x0[:, find(P[:, i])[0]].T, H, W)
    plt.title('cluster: %s' % str(i + 1))

plt.show()