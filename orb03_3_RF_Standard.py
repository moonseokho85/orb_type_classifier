# 패키지 임포트
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

feat_labels = train_x.columns[:]
# print(feat_labels)

# 훈련 데이터와 테스트 데이터로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

print(X_train.shape) # (159992, 21)
print(y_train.shape) # (159992,)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델링 / 훈련
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(X_train_scaled, y_train)

# 정확도 측정
acc = forest.score(X_test_scaled, y_test)
print('acc: ', acc) # 0.8803970099252482

# 예측
y_pred = forest.predict_proba(test_x)
# print(y_pred)

# 특성 중요도 그리기
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_orb(model):
    n_features = train_x.shape[1]
    plt.barh(np.arange(n_features), forest.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feat_labels)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
    
plot_feature_importances_orb(forest)
plt.show()

# # 제출 파일 생성
# submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
# submission.to_csv('submission.csv', index=True)
