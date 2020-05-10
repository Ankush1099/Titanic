#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset 
dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('test.csv')
X_train = dataset.iloc[:, [2,4,5]].values
y_train = dataset.iloc[:, 1].values
X_test = dataset1.iloc[:, [1,3,4]].values

#Dealing with missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X_train[:, [2]] = imputer.fit_transform((X_train[:, [2]]))
X_test[:, [2]] = imputer.fit_transform((X_test[:, [2]]))

#Encoding the Sex variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[:, 1] = le.fit_transform(X_train[:, 1])
X_test[:, 1] = le.fit_transform(X_test[:, 1])

#fitting the model
from sklearn.ensemble import RandomForestClassifier
cl = RandomForestClassifier(n_estimators = 300, criterion = 'entropy',random_state = 0)
cl.fit(X_train, y_train)

#make predictions
y_pred = cl.predict(X_test)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cl, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
'''
#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [500,800,1500,2500,5000], 'max_features': ['auto','sqrt','log2'], 
               'max_depth': [10,20,30,40,50], 'min_samples_split': [2,5,10,15,20],
                'min_samples_leaf': [1,2,5,10,15]}]
grid_search = GridSearchCV(estimator = cl,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''

#DataFrame
submission = pd.DataFrame({'PassengerId' : dataset1['PassengerId'], 'Survived' : y_pred})

#Converting to csv file
filename = 'Titanic Predictions.csv'
submission.to_csv(filename, index = False)
print('Saved File; ' + filename)
