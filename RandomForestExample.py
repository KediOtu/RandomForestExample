import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

random_state = int(input()) #we will use it as random_state while generating training sets and test sets
n = int(input())
rows = []
for i in range(n):
    rows.append([float(a) for a in input().split()])

X = np.array(rows)
y = np.array([int(a) for a in input().split()])

#generating training sets and test sets:
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=random_state)

#generating the random forest with 5 decision trees
rf = RandomForestClassifier(n_estimators=5)
rf.fit(X_train,y_train)

pred = rf.predict(X_test)
print(pred)
