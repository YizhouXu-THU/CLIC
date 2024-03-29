import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


npz_data = np.load('./data/example.npz', allow_pickle=True)
X_train = npz_data['scenario']
y_train = npz_data['label']
n_estimators = [300, 350, 360, 370, 800]
learning_rate = [0.01, 0.02, 0.04, 0.07, 0.08, 0.09]
params2 = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
adaboost = GridSearchCV(estimator=AdaBoostRegressor(DecisionTreeRegressor(max_depth=8)), 
                        param_grid=params2, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
adaboost.fit(X_train, y_train)

print('best_params_:', adaboost.best_params_)
print('best_score_:', adaboost.best_score_)
