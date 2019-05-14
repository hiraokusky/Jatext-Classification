import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)

df = pd.read_excel('icpc.xlsx')

train_X = []
train_y = []
for a in df.values:
    age = a[1]
    gender = a[2]
    icpc = a[4]
    result = a[6]
    # print(age, gender, icpc, result)
    train_X.append([age, gender, icpc])
    train_y.append(result)

(train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 666)

clf = RandomForestClassifier(random_state=0, n_estimators=500)
clf = clf.fit(train_X, train_y)

pred = clf.predict(test_X)

fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
print(auc(fpr, tpr))
print(accuracy_score(pred, test_y))
