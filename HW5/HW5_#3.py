import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def generate_data(n):
    a1 = np.array([3,3]).reshape(1,2)
    a2 = np.array([3,-3]).reshape(1,2)

    x1 = np.random.standard_normal(n)
    x2 = np.random.standard_normal(n)
    X = np.array([x1, x2]).reshape(2,n)
    Z = np.random.standard_normal(1)[0]

    var1 = a1 @ X
    Y = 1/(1+np.exp(-1*var1)) + (a2 @ X)**2 + (0.3*Z)

    X = X.reshape(n,2)
    return(X,Y[0])


# weighted_decays = [.0001, .01, 1, 100]
hidden_units = range(1,21)
scores = []
for i in hidden_units:
    X_train, Y_train = generate_data(100)
    reg = MLPRegressor(max_iter=5000, hidden_layer_sizes=(i,)).fit(X_train, Y_train)

    X_test, Y_test = generate_data(1000)
    scores.append(reg.score(X_test, Y_test))


    # plt.plot(reg.loss_curve_)
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE')
    # plt.title(f'Hidden Units: {i}')
    # plt.show()

plt.plot(hidden_units, scores)
plt.xlabel('Hidden Units')
plt.ylabel('Score')
plt.show()
