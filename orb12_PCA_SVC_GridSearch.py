# 패키지 임포트
import pandas as pd

# 데이터 불러오기
train = pd.read_csv('./data/train.csv', index_col=0)
test = pd.read_csv('./data/test.csv', index_col=0)
sample_submission = pd.read_csv('./data/sample_submission.csv', index_col=0)

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

feat_labels = train_x.columns[:]
print(feat_labels)

# 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x_scaled = scaler.transform(train_x)
print(train_x_scaled)

# 차원 축소
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(train_x_scaled)
train_x_pca = pca.transform(train_x_scaled)
print("원본 데이터 형태: ", train_x_scaled.shape)
print("축소된 데이터 형태: ", train_x_pca.shape)

# feat_labels = train_x_pca.columns[:]
# print(feat_labels) # array라서 칼럼을 알 수 없음.

# 훈련 데이터와 테스트 데이터로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x_pca, train_y, test_size=0.2)

print(X_train.shape) # (159992, 21)
print(y_train.shape) # (159992,)

# Parameter
parameters = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma": [0.001, 0.0001]}
]

# K-Fold
from sklearn.model_selection import KFold
kfold_cv = KFold(n_splits=5, shuffle=True)

# GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
model = GridSearchCV(SVC(), parameters, cv=kfold_cv, n_jobs=-1)
model.fit(X_train, y_train)
print("Optimal parameter: ", model.best_estimator_)

# Evaluating in optimal parameter
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('Final accuracy: ', accuracy_score(y_test, y_pred))
