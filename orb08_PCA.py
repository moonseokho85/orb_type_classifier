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

print(train_x)

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
