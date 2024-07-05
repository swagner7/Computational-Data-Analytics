import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# load data
data = pd.read_csv('marriage.csv', header = None, index_col = False)
X = data.iloc[:,:-1] # split data into features (X)...
Y = data.iloc[:,-1] # ... and responses (Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # split into train (80%) and test (20%) sets

# initialize models for comparison
classifiers = {'NB': GaussianNB(var_smoothing=1e-3), 'Log': LogisticRegression(),'KNN': KNeighborsClassifier()}

def run_classification(model_name):
    model = classifiers[model_name]
    model.fit(X_train, Y_train) # fit model to data
    print(model_name + ' training accuracy: ', model.score(X_train, Y_train)) # print training accuracy score
    print(model_name + ' testing accuracy: ', model.score(X_test, Y_test)) # print testing accuracy score

# test classifiers
[run_classification(model_name) for model_name in classifiers]

# run PCA on data
print('============================')
pca = PCA(n_components=2) # initialize PCA
pca.fit(X_train) # fit PCA to data
X_train = pca.transform(X_train) # project data using PCA moel
X_test = pca.transform(X_test)

# re-test classifiers
[run_classification(model_name) for model_name in classifiers]

# plot decision boundary
x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min - 1, x_max + 1, 0.1), np.arange(y_min - 1, y_max + 1, 0.1))
f, axarr = plt.subplots(1, 3, figsize=(12,4))

for idx, clf, tt in zip([0, 1, 2], list(classifiers.values()), list(classifiers.keys())):
    clf.fit(X_train, Y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx].contourf(xx, yy, Z, alpha = 0.5)
    for i in range(2):
        axarr[idx].scatter(X_train[Y_train == i, 0], X_train[Y_train == i, 1], label=str(i))

    axarr[idx].legend()
    axarr[idx].set_title(tt)

plt.show()