from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

##########데이터 로드

x_data = [
[2, 1],
[3, 2],
[3, 4],
[5, 5],
[7, 5],
[2, 5],
[8, 9],
[9, 10],
[6, 12],
[9, 2],
[6, 10],
[2, 4]
]
y_data = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0]

labels = ['fail', 'pass']

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 학습

model = LogisticRegression()

model.fit(x_train, y_train)

##########모델 검증

y_predict = model.predict(x_test)

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True)
plt.show()