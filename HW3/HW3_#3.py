import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io
from scipy.stats import multivariate_normal as mvn
import scipy.sparse.linalg as ll
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.metrics import precision_score
from sklearn.cluster import KMeans

np.set_printoptions(threshold=sys.maxsize)

# load data
data = scipy.io.loadmat('data.mat')['data']
label = scipy.io.loadmat('label.mat')['trueLabel']

data = np.array(data).T  # transpose data and cast to array
label = np.array(label) # cast labels to array

mu_0 = np.mean(data, axis=0, keepdims=True)
norm_data = data - mu_0
m, n = norm_data.shape
C = np.matmul(norm_data.T, norm_data) / m

# implement pca and project to 2 dimensions
d = 5
V, g, _ = np.linalg.svd(C) # singular value decomp
V = V[:, :d]
g = np.diag(g[:d])
proj_data = np.dot(norm_data, V)

# implement EM-GMM algorithm
K = 2 # number of mixtures
np.random.seed(5)
pi = np.random.random(K)
pi = pi / np.sum(pi)

# initial mean and covariance
mu = np.random.randn(K, d)
mu_old = mu.copy()

sigma = [] # list storage for sigmas
for ii in range(K):
    seed = 1 if ii == 0 else 4
    np.random.seed(5)
    dummy = np.random.randn(d, d)
    sigma.append(dummy @ dummy.T + np.eye(d))

# initialize the posterior
t = np.full((m, K), fill_value=0.)

iter_max = 120
tol = 0.001

# implement EM algorithm
log_likelihood = [] # storage for log_likelihoods
for i in range(iter_max):

    # E-step
    for j in range(K):
        sigma_det = np.linalg.det(sigma[j])
        t[:, j] = pi[j] * mvn.pdf(proj_data, mu[j], sigma[j])

    sum_t = np.sum(t, axis=1)
    sum_t.shape = (m, 1)
    t = np.divide(t, np.tile(sum_t, (1, K)))

    log_likelihood.append(np.sum(np.log(sum_t)))

    # M-step
    for j in range(K):
        # update prior
        pi[j] = np.sum(t[:, j]) / m

        # update component mean
        mu[j] = proj_data.T @ t[:, j] / np.sum(t[:, j], axis=0)

        # update cov matrix
        dummy = proj_data - np.tile(mu[j], (m, 1))  # X-mu
        sigma[j] = dummy.T @ np.diag(t[:, j]) @ dummy / np.sum(t[:, j], axis=0)

    if np.linalg.norm(mu - mu_old) < tol:
        break

    mu_old = mu.copy()

    if i == iter_max-1:
        print('Max Iterations!')
        break


plt.figure()
plt.plot(log_likelihood,'-x')
plt.title('Log-Likelihood Convergence')
plt.ylabel('Log-Likelihood')
plt.xlabel('Iteration')
plt.show()

mean_1 = (V @ mu[0] + mu_0).reshape((28,28)).T
mean_2 = (V @ mu[1] + mu_0).reshape((28,28)).T
covariance_1 = V @ np.sqrt(g) @ sigma[0] @ np.sqrt(g) @ V.T
covariance_2 = V @ np.sqrt(g) @ sigma[1] @ np.sqrt(g) @ V.T

plt.imshow(mean_1)
plt.title('Mean image of component #1 | Weight = ' + str(round(pi[0], 2)))
plt.show()

plt.imshow(mean_2)
plt.title('Mean image of component #2 | Weight = ' + str(round(pi[1], 2)))
plt.show()

plt.imshow(covariance_1)
plt.title('Covariance matrix of component #1')
plt.show()

plt.imshow(covariance_2)
plt.title('Covariance matrix of component #2')
plt.show()

# implement K-means clustering for comparison
kmeans = KMeans(n_clusters=2, random_state=0).fit(proj_data)

label = label[0]
label[label==2] = 0
label[label==6] = 1

classification = np.argmin(t, axis=1)

print(f"\nGMM precision score: {precision_score(label,classification)}")

print(classification_report(label, classification, target_names=['digit 2', 'digit 6']))

confusion_df = pd.DataFrame(confusion_matrix(label,classification),
             columns=["Predicted Class " + str(x) for x in [0,1]],
             index = ["Class " + str(x) for x in [0,1]])

print(confusion_df)
err_rate2 = np.mean(label[0:1032] != classification[0:1032])
err_rate6 = np.mean(label[1032:] != classification[1032:])
print("\nGMM mis-classification rate for digit 2:", err_rate2)
print("GMM mis-classification rate for digit 6:", err_rate6)

print("\nkmeans precision score: {}".format(precision_score(label,kmeans.labels_)))

print(classification_report(label, kmeans.labels_, target_names=['digit 2', 'digit 6']))

confusion_df = pd.DataFrame(confusion_matrix(label,kmeans.labels_),
             columns=["Predicted Class " + str(x) for x in [0,1]],
             index = ["Class " + str(x) for x in [0,1]])

print(confusion_df)
err_rate2 = np.mean(label[0:1032] != kmeans.labels_[0:1032])
err_rate6 = np.mean(label[1032:] != kmeans.labels_[1032:])
print("\nkmeans mis-classification rate for digit 2:", err_rate2)
print("kmeans mis-classification rate for digit 6:", err_rate6)