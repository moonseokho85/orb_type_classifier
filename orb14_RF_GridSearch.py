# 패키지 임포트
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 데이터 불러오기
train = pd.read_csv('../Data/orb_data/train.csv', index_col=0)
test = pd.read_csv('../Data/orb_data/test.csv', index_col=0)
sample_submission = pd.read_csv('../Data/orb_data/sample_submission.csv', index_col=0)

# Train 데이터의 타입을 Sample_submission에 대응하는 가변수 형태로 변환
column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train['type_num'] = train['type'].apply(lambda x : to_number(x, column_number))

# 모델에 적용할 데이터 셋 준비 
train_x = train.drop(columns=['type', 'type_num'], axis=1)
train_y = train['type_num']
test_x = test

# print(train_x)
# print(train_y)

# 특성 라벨 정하기
feat_labels = train_x.columns[:]
# print(feat_labels)

# 훈련 데이터와 테스트 데이터로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

# Parameter
param_grid = {
    'n_estimators': [400, 700, 1000],
    'max_depth': [15,20,25],
    'max_leaf_nodes': [50, 100, 200]
}

# K-Fold
from sklearn.model_selection import KFold
kfold_cv = KFold(n_splits=5, shuffle=True)

# GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
model = GridSearchCV(RandomForestClassifier(), param_grid, cv=kfold_cv, n_jobs=-1)
model.fit(X_train, y_train)
print("Optimal parameter: ", model.best_estimator_)
'''
Optimal parameter:  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=200,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
'''

# Evaluating in optimal parameter
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('Final accuracy: ', accuracy_score(y_test, y_pred)) # Final accuracy:  0.8476961924048101

'''
# # 제출 파일 생성
# submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
# submission.to_csv('submission.csv', index=True)
'''
