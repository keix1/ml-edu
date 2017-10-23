from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # ガクの幅と花びらの幅
y = iris.target
# print("class labels:", np.unique(y))
# print('X', len(X))
# print(X)
# print('y', len(y))
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

# transformは標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(n_iter=1000000, eta0=0.5, random_state=0, shuffle=True)
# n_iterはエポック数（データセットのトレーニング回数）
# eta0は学習率

ppn.fit(X_train_std, y_train) # fitで学習

y_pred = ppn.predict(X_test_std) # アルゴリズム.predictで予測

print('X_trainはデータのリスト')
print(X_train)
print('y_trainはラベルのリスト')
print(y_train)
print('y_predは予測したラベルのリスト')
print(y_pred)

print('Misclassfied samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# lr = LogisticRegression(C=1000.0, random_state=0)
# lr.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [stardized]')
# plt.ylabel('petal width [stardized]')
# plt.legend(loc='upper left')
# plt.show()
