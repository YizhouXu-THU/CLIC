import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

eval_size = 4096
npz_data = np.load('./data/example.npz', allow_pickle=True)
X = npz_data['scenario']
y = npz_data['label']
total_num, max_dim = X.shape
test_size = total_num - eval_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

pred = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=350, learning_rate=0.07)
pred.fit(X_train, y_train)

y_score = pred.predict(X_test)
loss = log_loss(y_test, y_score)
y_pred = pred.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Loss:', loss, 'Accuracy:', accuracy)
