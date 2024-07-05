import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

def read_data(filename):
    file = open(filename, "r") # open file
    data = [] # initialize data storage
    for line in file: # go through each line
        line = line.strip().split(',') # parse out data
        line = list(map(float ,line))
        data.append(line) # append to storage

    return(pd.DataFrame(data))

def read_names(filename):
    file = open(filename ,'r')
    names = []
    start_read = False
    for line in file:
        line = line.strip()
        if line and start_read:
            names.append(line.split(':')[0])
        if line.startswith('1, 0'):
            start_read = True
    names.append('spam')

    return(names)

# read data
filename = 'spambase/spambase.data'
df = read_data(filename)
df.columns = read_names('spambase/spambase.names')

# build X and y data sets
X = df.drop('spam', axis=1)
y = df['spam']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17990, shuffle=True)

# build a decision tree classifier
dt_model = tree.DecisionTreeClassifier(random_state=11389, max_leaf_nodes=10) # initialize module
dt_model.fit(X_train,y_train) # fit to data
dt_accuracy = dt_model.score(X_test,y_test) # return accuracy score
tree.plot_tree(dt_model, filled=True, feature_names = df.columns[:-1]) # plot tree
print('Decision Tree Classifier error: ', 1 - dt_accuracy)

# random forest classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train,y_train)
rf_accuracy = rf_model.score(X_test,y_test)
print('Random Forest Classifier error: ', 1 -  rf_accuracy)

# evaluate relationship between error and depth
test_errors = []
for num_trees in range(1,100,5):
    model = RandomForestClassifier(n_estimators = num_trees, random_state=11)
    model.fit(X_train, y_train)
    score = model.score(X_test,y_test)
    test_errors.append([num_trees, 1 - score])

test_errors = np.array(test_errors)
tree_test_error = [1 - tree_accuracy] * len(test_errors)

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(test_errors[:,0], test_errors[:,1], label = 'Random Forest')
ax.plot(test_errors[:,0], tree_test_error, label = 'CART')
plt.legend()
plt.title('Comparison of Test Error for Decision Tree and Random Forest Classification')
plt.xlabel('Number of Trees')
plt.ylabel('Test Error')
plt.show()

#----------------------
test_errors = []
oob_errors = []
vs = []
for v in range(1,30,2):
    model = RandomForestClassifier(min_samples_split=v, random_state=11, bootstrap=True, oob_score=True)
    model.fit(X_train, y_train)
    score = model.score(X_test,y_test)
    test_errors.append(1 - score)
    oob_errors.append(1 - model.oob_score_)
    vs.append(v)

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(vs, test_errors, label = 'Test Error')
ax.plot(vs, oob_errors, label = 'OOB Error')
plt.legend()
plt.title('Comparison of Error Types in Random Forest Classification')
plt.xlabel('Min_Samples_Leaf (v)')
plt.ylabel('Error')
plt.show()


## One-class SVM
X_train_nonspam = X_train[y_train == 0] # extract nonspam emails from the training data

clf = OneClassSVM(kernel = 'rbf').fit(X_train_nonspam) # train a one-class SVM using RBF kernel
y_test_pred = clf.predict(X_test)

# inliers are nonspam emails
y_test_pred[y_test_pred == 1] = 0
# outliers are spam emails
y_test_pred[y_test_pred == -1] = 1

one_class_svm_test_error = sum(y_test_pred != y_test)/len(y_test)
print('One-Class SVM Model error: ',one_class_svm_test_error)