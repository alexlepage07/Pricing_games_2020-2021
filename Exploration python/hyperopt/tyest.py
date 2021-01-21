from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regressor = RandomForestRegressor(max_depth=2, random_state=0)
regressor.fit(X, y)
print(regressor.predict(X))



