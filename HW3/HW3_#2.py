import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy import stats
from sklearn.neighbors import KernelDensity


df = pd.read_csv("n90pol.csv")

# plot historgrams
data = df.to_numpy()
data = preprocessing.scale(data)

amygdala = data[:,0]
acc = data[:,1]

bins = 10

plt.hist(amygdala, bins, alpha=0.5, label='amygdala')
plt.hist(acc, bins, alpha=0.5, label='acc')
plt.legend(loc='upper right')
plt.title(f'1D Histogram w/ {bins} bins')
plt.show()

# plot kdes
kde = df[["amygdala", "acc"]].plot.kde()
plt.title("KDE")
plt.show()

# plot 2D histogram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, x_edg, y_edg = np.histogram2d(amygdala, acc, bins=bins)
x_pos, y_pos = np.meshgrid(x_edg[:-1] + x_edg[1:], y_edg[:-1] + y_edg[1:])
x_pos = x_pos.flatten()/2.
y_pos = y_pos.flatten()/2.
z_pos = np.zeros_like(x_pos)
ax.bar3d(x_pos, y_pos, z_pos, x_edg[1] - x_edg[0], y_edg[1] - y_edg[0], hist.flatten())
ax.set_title(f'2D Histogram w/ {bins} bins')
ax.set_xlabel('amygdala')
ax.set_ylabel('acc')
plt.show()

# plot 2D kde (using scipy)
min_x, max_x = np.min(np.array(df.amygdala)), np.max(np.array(df.amygdala))
min_y, max_y = np.min(np.array(df.acc)), np.max(np.array(df.acc))
X, Y = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
Z = np.reshape(stats.gaussian_kde(np.vstack([np.array(df.amygdala), np.array(df.acc)]))(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)

fig = plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_title('2D KDE Contour Plot')
ax.set_xlabel('amygdala')
ax.set_ylabel('acc')
plt.show()

amygdala = np.array(df.amygdala)
acc = np.array(df.acc)
orientation = np.array(df.orientation)

# conditional distributions
def one_D_kde(data,grids):
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data[:, None])
    lp = kde.score_samples(grids[:, None])
    return np.exp(lp)

def conditional_kde_plot(data,mesh,label):
    for i in range(2,6):
        kde = one_D_kde(data[orientation == i],mesh)
        plt.plot(mesh, kde)
        plt.title(f'P({label} | orientation = {str(i)})')
        plt.xlabel(label)
        plt.ylabel('density')
        plt.show()

# joint conditional distributions
def two_D_KDE(data, mesh):
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data)
    logprob = kde.score_samples(mesh.T)
    joint = np.reshape(np.exp(logprob), X.shape)
    return joint

def conditonal_joint_kde_plot(data, mesh):
    for i in range(2,6):
        joint_kde = two_D_KDE(data[orientation == i],mesh)
        fig = plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, joint_kde)
        plt.title(f'P(amygdala, acc | orientation = {str(i)})')
        plt.xlabel('amygdala')
        plt.ylabel('acc')
        plt.show()

conditional_kde_plot(amygdala, X[:,0], 'amygdala')
conditional_kde_plot(acc, Y[0], 'acc')
conditonal_joint_kde_plot(np.array(df[['amygdala','acc']]),np.vstack([X.ravel(), Y.ravel()]))



