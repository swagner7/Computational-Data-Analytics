import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt

# load data --------------------------------------
raw_data = pd.read_csv('food-consumption.csv') # read data from csv
data = np.array(raw_data[raw_data.columns[1:]]) # convert pd dataframe into np array
rows = data.shape[0] # initialize rows constant for looping
cols = data.shape[1] # initialize columns constant for looping

# run PCA --------------------------
def run_PCA(data):
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

# draw 2D PCA scatterplots ------------------------
PC_1, PC_2 = run_PCA(data)
fig, ax = plt.subplots()
ax.scatter(PC_1, PC_2) # generate scatterplot

country_labels = raw_data['Country']
for i in range(len(PC_1)): # add labels to each point
    ax.annotate(country_labels[i], (PC_1[i], PC_2[i]))

plt.title("2D European Food Consumption PCA Scatterplot") # add titles
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

PC_1, PC_2 = run_PCA(data.T)
fig, ax = plt.subplots()
ax.scatter(PC_1, PC_2) # generate scatterplot

food_labels = raw_data.columns[1:]
for i in range(len(PC_1)): # add labels to each point
    ax.annotate(food_labels[i], (PC_1[i], PC_2[i]))

plt.title("2D European Food Consumption PCA Scatterplot") # add titles
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
