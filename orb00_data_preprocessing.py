import pandas as pd

# 데이터 불러오기
train = pd.read_csv('../Data/orb_data/train.csv', index_col=0)
test = pd.read_csv('../Data/orb_data/test.csv', index_col=0)
sample_submission = pd.read_csv('../Data/orb_data/sample_submission.csv', index_col=0)

# 데이터 전처리
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

# 상관관계: -0.84 (petroMag_u, psfMag_u), (petroMag_u, fiberMag_u)

# 이상치 감지 객체 만들기
from sklearn.covariance import EllipticEnvelope
outlier_detector = EllipticEnvelope(contamination=.1)

# 감지 객체 훈련
outlier_detector.fit(train_x)

# 이상치 예측
pred = outlier_detector.predict(train_x)
print(pred)
print(pred.shape)

import numpy as np
def find_idx(x):
    return np.where((x < 0))

idx = find_idx(pred)
print(idx)
print(len(idx))
print(idx[0])
print(idx[0].shape)

# 이상치의 인덱스를 반환하는 함수
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

