import pandas as pd

# 데이터 불러오기
train = pd.read_csv('../Data/orb_data/train.csv', index_col=0)
test = pd.read_csv('../Data/orb_data/test.csv', index_col=0)
sample_submission = pd.read_csv('../Data/orb_data/sample_submission.csv', index_col=0)

# 데이터 요약
print(train.describe())
'''
             fiberID      psfMag_u       psfMag_g       psfMag_r       psfMag_i       psfMag_z  ...     petroMag_z     modelMag_u     modelMag_g     modelMag_r     modelMag_i     modelMag_z
count  199991.000000  1.999910e+05  199991.000000  199991.000000  199991.000000  199991.000000  ...  199991.000000  199991.000000  199991.000000  199991.000000  199991.000000  199991.000000
mean      360.830152 -6.750146e+00      18.675373      18.401235      18.043495      17.663526  ...      17.699207      20.110991      18.544375      18.181544      17.692395      17.189281
std       225.305890  1.187678e+04     155.423024     127.128078     116.622194     123.735298  ...     142.691880     122.299062     161.728183     133.984475     131.183416     133.685138
min         1.000000 -5.310802e+06  -40022.466071  -27184.795793  -26566.310827  -24878.828280  ...  -30070.729379  -26236.578659  -36902.402336  -36439.638493  -38969.416822  -26050.710196
25%       174.000000  1.965259e+01      18.701180      18.048572      17.747663      17.425523  ...      16.804705      19.266214      18.076120      17.423425      16.977671      16.705774
50%       349.000000  2.087136e+01      19.904235      19.454492      19.043895      18.611799  ...      18.174592      20.406840      19.547674      19.143156      18.641756      18.100997
75%       526.000000  2.216043e+01      21.150297      20.515936      20.073528      19.883760  ...      19.807652      21.992898      20.962386      20.408140      19.968846      19.819554
max      1000.000000  1.877392e+04    3538.984910    3048.110913    4835.218639    9823.740407  ...   17403.789263   14488.251976   10582.058590   12237.951703    4062.499371    7420.534172
'''
# 상관관계 분석
print(train.corr())

# 히트맵 그리기
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 15))
sns.heatmap(data=train.corr(), annot=True)
plt.show() # 상관관계: -0.84 (petroMag_u, psfMag_u), (petroMag_u, fiberMag_u)

'''
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

print(type(train_x))
print("Shape of orb data: ", train_x.shape)

print(train_x.columns)
'''

'''
Index(['fiberID', 'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z', 'fiberMag_u', 'fiberMag_g', 'fiberMag_r', 'fiberMag_i', 'fiberMag_z', 'petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z', 'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z'], dtype='object')
'''

'''
X_psfMag_u = train_x['psfMag_u']
print(X_psfMag_u)
print(X_psfMag_u.shape)

import numpy as np
X_psfMag_u = np.reshape(X_psfMag_u, (-1, 1))
print(X_psfMag_u.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_psfMag_u)
X_psfMag_u_scaled = scaler.transform(X_psfMag_u)
print(X_psfMag_u_scaled)
'''