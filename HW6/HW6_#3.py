from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.6, 0.75]).reshape(-1,1)
Y = np.array([0, 1]).reshape(-1,1)

test_sample = 0.75
X_test = np.array([test_sample, test_sample]).reshape(-1,1)

ridge_model = Ridge()
mses = []
alphas = np.arange(0.01, 2, 0.01)

for a in alphas:
    ridge_model.set_params(alpha = a)
    ridge_model.fit(X,Y)
    mses.append(mean_squared_error(X, ridge_model.predict(X_test)))


plt.plot(alphas, mses)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()


test_sample = 0.1
X_test = np.array([test_sample, test_sample]).reshape(-1,1)

ridge_model = Ridge()
mses = []
alphas = np.arange(0.01, 2, 0.01)

for a in alphas:
    ridge_model.set_params(alpha = a)
    ridge_model.fit(X,Y)
    mses.append(mean_squared_error(X, ridge_model.predict(X_test)))


plt.plot(alphas, mses)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()