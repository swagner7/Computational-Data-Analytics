import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
import random
import scipy.io


### MNIST data set
# load data
data = scipy.io.loadmat('data/mnist_10digits.mat')
X_train, y_train, X_test, y_test = data['xtrain'], data['ytrain'].T, data['xtest'], data['ytest'].T

y_train = y_train[:,0]
y_test = y_test[:,0]

# downsampling the training data
random.seed(1)
indices = random.sample(range(60000), 5000)
indices_median_trick = random.sample(range(60000), 1000)

# sample for median trick
X_sample = X_train[indices]/255

# downsampling the training data
X_train = X_train[indices]
y_train = y_train[indices]

# standardize the features
X_train = X_train/255
X_test = X_test/255

# Build KNN model
print('Tuning KNN model n_neighbors...')
param_grid = {
    'n_neighbors': range(1, 10)
}

gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, n_jobs = -1, verbose = 0)
best_model = gs.fit(X_train,y_train)

print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

knn = best_model.best_estimator_
print(knn)

# Build Logistic Regression model
print('\nBuilding Logistic Regression model...')
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
print(logreg)

# Build SVC model
print('\nBuilding Support Vector Classifier model')
svc = LinearSVC()
svc.fit(X_train,y_train)
print(svc)

# Build kernel SVC model
print('Building Kernel Support Vector Classifier model')
kernel_SVC = SVC(kernel = 'rbf')
kernel_SVC.fit(X_train, y_train)
print(kernel_SVC)

# Build Neural Network model
print('\nBuilding Neural Network model')
NN = MLPClassifier(hidden_layer_sizes = (20,10))
NN.fit(X_train,y_train)
print(NN)


# Predict on test data
classifiers = { 'KNN classifier': knn,
        'Logistic Regressor': logreg,
        'SVM': svc,
        'Kernel SVM': kernel_SVC,
        'Neural Network': NN
         }

for model_name in classifiers:
    model = classifiers[model_name]
    y_pred = model.predict(X_test)
    print(model_name + ':')
    print(classification_report(y_test, y_pred))
    print('confusion matrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('*****************************************************')