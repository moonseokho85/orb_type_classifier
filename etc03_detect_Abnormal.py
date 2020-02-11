# 라이브러리를 임포트
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# 모의 데이터 만들기
features, _ = make_blobs(n_samples=10, n_features=2, centers=1, random_state=1)
'''
print(features)
print(features.shape) # (10, 2)
'''
# 첫번째 샘플을 극단적인 값으로 바꾸기
features[0, 0] = 10000
features[0, 1] = 10000

# print(features)

# 이상치 감지 객체를 만들기
outlier_detector = EllipticEnvelope(contamination=.1)

# 감지 객체를 훈련
outlier_detector.fit(features)

# 이상치를 예측
pred = outlier_detector.predict(features)
print(pred)

# 하나의 특성을 만들기
feature1 = features[:, 0]
feature2 = features[:, 1]

# 이상치의 인덱스를 반환하는 함수
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# 함수 실행
idx1 = indicies_of_outliers(feature1)
idx2 = indicies_of_outliers(feature2)
print(idx1)
print(idx2)

print(feature1[0])