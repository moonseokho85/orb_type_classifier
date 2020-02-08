import pandas as pd

#### Loading Data ####
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# print("train_Dataframe: ", train_data)
print(train_data.shape) # (199991, 23)

# print("test_DataFrame: ", test_data)
print(test_data.shape) # (10009, 22)

import pandas_profiling
pr = train_data.profile_report()
pr.to_file('./pr_report.html') # pr_report.html 파일로 저장

#### Data preprocessing ####
y = train_data['type']
x = train_data.drop('type', axis=1)

label = set(y)
print(label)

#### Split Data to Train and Test set ####
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

'''
#### RandomForestClassifier ####
# Making model (RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Fitting Model
model.fit(x_train, y_train)

# Scoring Model
aaa = model.score(x_test, y_test)
print("aaa: ", aaa) # 0.8673716842921073
'''

'''
#### K-Fold + GridSearch + RandomForestClassifier ####
# Parameter
param_grid = [
    {"n_estimators": [1, 10, 100, 1000]}
]

# K-Fold
from sklearn.model_selection import KFold
kfold_cv = KFold(n_splits=5, shuffle=True)

# GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
model = GridSearchCV(RandomForestClassifier(), param_grid, cv=kfold_cv)
model.fit(x_train, y_train)
print("Optimal parameter: ", model.best_estimator_)

# Evaluating in optimal parameter
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print('Final accuracy: ', accuracy_score(y_test, y_pred))
'''

'''
#### K-Fold + RandomizedSearchCV + RandomForestClassifier ####
# Parameter
param_grid = {
    "n_estimators": [1, 10, 100, 1000]
    }

# K-Fold
from sklearn.model_selection import KFold
kfold_cv = KFold(n_splits=5, shuffle=True)

# GridSearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
model = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=kfold_cv)
model.fit(x_train, y_train)
print("Optimal parameter: ", model.best_estimator_)

# Evaluating in optimal parameter
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print('Final accuracy: ', accuracy_score(y_test, y_pred))
'''
'''
#### All algorithms ####
# Extracting All classifier algorithm
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils.testing import all_estimators
allAlgorithms = all_estimators(type_filter="classifier")

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    
    # Producing each algorithms
    clf = algorithm()
    
    # Fitting and predicting
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    # Scoring models
    from sklearn.metrics import accuracy_score
    print(name, "의 정답률: ", accuracy_score(y_test, y_pred))
'''

'''
#### Feature importance & PCA ####
from sklearn import decomposition

dmodel = decomposition.PCA(n_components=0.97) # Explains 97% of total variance

train_dt = df_train.copy()
dmodel.fit(train_dt)
train_d = dmodel.transform(train_dt)

print(dmodel.explained_variance_ratio_)
'''