import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy.linalg import svd

# downsample an image by given factor
def downsample(image, x):

    w = image.shape[1]
    h = image.shape[0]
    downscaled_image = np.zeros(shape=(h // x, w // x), dtype=np.uint8)

    for i in range(0, h-x, x):
        for j in range(0, w-x, x):
           downscaled_image[i//x, j//x] = np.mean(image[i:(i + x), j:(j + x)], axis=(0, 1))

    downscaled_image = downscaled_image.astype(np.uint8)

    return downscaled_image

# run PCA
def run_PCA(matrix, n):

    matrix = matrix.T # transpose matrix
    matrix = matrix - np.mean(matrix, axis = 1, keepdims = True) # subtract means
    W, _, _, = svd(matrix) # singular value decomposition
    W = W[:,0:n] # eigenvectors/eigenfaces for the first n

    return W

# plot eigenfaces
def plot_eigenfaces(W):

    fig, axs = plt.subplots(1, W.shape[1], figsize=(16,3))
    axs = axs.ravel()
    for i in range(W.shape[1]):
        image = W[:,i].reshape((60,80))
        axs[i].imshow(image)

    plt.show()

# perform face recognition
def similarity(a,b):
    return np.sum(a*b)/(np.linalg.norm(a) * np.linalg.norm(b))


subject01 = [] # initialize lists to hold images for each subject
subject02 = []

for file_name in glob.glob("yalefaces/*"): # for each file in Yale Faces
    image = plt.imread(file_name) # read in image

    resized_image = downsample(image, 4).flatten() # project and downsample by a factor of 4
    if 'subject01' in file_name: # sort images by subject
        subject01.append(resized_image)
    else:
        subject02.append(resized_image)

test_subj1 = subject01[-1]
test_subj2 = subject02[-1]
subject01 = np.array(subject01)[:-1] # remove the test image
subject02 = np.array(subject02)[:-1]

# compute and visualize n eigenfaces
W1 = run_PCA(subject01, 6)
plot_eigenfaces(W1)

W2 = run_PCA(subject02, 6)
plot_eigenfaces(W2)

# print the similarity scores
print(similarity(W1[:,0],test_subj1))
print(similarity(W2[:,0],test_subj1))
print(similarity(W1[:,0],test_subj2))
print(similarity(W2[:,0],test_subj2))