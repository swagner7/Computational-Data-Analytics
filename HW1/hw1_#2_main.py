import imageio
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import time

k = 8 # input # of desired clusters
k0 = k # duplicate k that won't change (for illustration at end)
image_name = 'football.bmp' # input image

print(f'\nStarting k-means image compression with {k} clusters...\n')
print(f'Reading {image_name} image...\n')

filepath = os.path.join(os.getcwd(), image_name) # get filepath to the image
image = imageio.v2.imread(filepath) # read in image

print('Choosing random initial centroids...')
centroids = np.random.rand(k, 3)*255 # randomly choose initial centroids
print(centroids)
cluster_assignments = np.zeros((image[:,:,0].shape), dtype = np.uint8) # create array to store cluster assignments

t0 = time.time() # initialize timer
movements = [1000, 999]
iters = 0
while (movements[-2] - movements[-1]) > 0.0: # while the centroids are still being moved
    # assign points to clusters
    print('\nAssigning points to clusters...\n')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]): # for each point

            norm_true = 1000.0
            for x in range(0, centroids.shape[0]): # try each centroid
                norm = np.linalg.norm(image[i, j, :] - centroids[x], 2) # calculate l2 norm
                if norm < norm_true: # find shortest distance to assign point to that cluster
                    cluster_assignments[i,j] = x
                    norm_true = norm

    # calculate new centroids
    print('Calculating new clusters...\n')
    new_centroids = np.zeros((k, 3)) # create storage for new centers
    count = np.zeros(k)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]): # for each point
            new_centroids[cluster_assignments[i,j]] += image[i,j,:] # total up
            count[cluster_assignments[i,j]] += 1 # count up

    if (np.any(count == 0.0)):
        indexes = np.where(count == 0)[0]
        for i in range(len(indexes)):
            k -= 1
            new_centroids = np.delete(new_centroids, indexes[-i - 1], 0)
            count = np.delete(count, indexes[-i - 1], 0)
            cluster_assignments -= (cluster_assignments >= indexes[-i - 1]).astype(np.uint8)

    for x in range(k):
        new_centroids[x] = np.divide(new_centroids[x], count[x]) # take average

    print('Moving centroids...')
    # calculate distance centroid moved
    deltas = []
    for x in range(k):
        delta = np.linalg.norm(new_centroids[x] - centroids[x], 2)
        deltas.append(delta)

    centroid_movement = sum(deltas)/len(deltas)
    movements.append(centroid_movement)
    print('Movement: ', centroid_movement)
    iters += 1

    centroids = new_centroids # reassign centers for next loop
t1 = time.time()
print('Time elapsed: ', t1-t0)
print('Iterations: ', iters)
print('TpI: ', (t1-t0)/iters)

f, axarr = plt.subplots(1, 2)
f.suptitle(f'Image Comparison (Clusters: {k0})')
axarr[0].set_title('Original Image')
axarr[0].imshow(image)
axarr[1].set_title(f'K-Means Compressed Image with {k0} Clusters')
axarr[1].imshow(cluster_assignments)
plt.show()



