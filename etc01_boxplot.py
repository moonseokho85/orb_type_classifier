import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xs = np.array(np.linspace(start=0, stop=100, num=101)) # 시작, 끝(포함), 갯수
print(xs)

df = pd.DataFrame(xs, columns=['feature1'])
print(df)

plt.figure(figsize=(7, 6))
boxplot = df.boxplot(column=['feature1'])
plt.yticks(np.arange(0, 101, step=5))
plt.show()