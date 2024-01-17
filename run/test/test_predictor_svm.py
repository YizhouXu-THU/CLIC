import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

eval_size = 4096
npz_data = np.load('./data/example.npz', allow_pickle=True)
X = npz_data['scenario']
y = npz_data['label']
total_num, max_dim = X.shape
test_size = total_num - eval_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

clf = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True)
clf.fit(X_train, y_train)

y_score = clf.predict_proba(X_test)
loss = log_loss(y_test, y_score)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Loss:', loss, 'Accuracy:', accuracy)
