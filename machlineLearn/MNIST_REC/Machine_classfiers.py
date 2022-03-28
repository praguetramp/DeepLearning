import pandas as pd

data = pd.read_csv('train.csv')
y = data['label']
data.drop('label', axis=1, inplace=True)
X = data
y = pd.Categorical(y)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

logreg = LogisticRegression()
dt = DecisionTreeClassifier()
svc = LinearSVC()

model_logred = logreg.fit(X, y)
model_dt = dt.fit(X, y)
model_svc = svc.fit(X, y)

X_test = pd.read_csv('test.csv')

pred_logreg = model_logred.predict(X_test)
pred_dt = model_dt.predict(X_test)
pred_svc = model_svc.predict(X_test)

from keras import metrics

pred1 = model_logred.predict(X)
pred2 = model_dt.predict(X)
pred3 = model_svc.predict(X)

print("Decision Tree Accuracy is :", metrics.accuracy(pred_dt, y))
print("Logistics Regression Accuracy is :", metrics.accuracy(pred_logreg, y))
print("Support Vector Machine Accuracy is :", metrics.accuracy(pred_svc, y))
