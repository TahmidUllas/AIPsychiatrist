import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = pd.read_csv('anxiety_data.csv')
#data =pd.read_csv('depression_data.csv')
#data= pd.read_csv('hopelessness_data.csv')

df = pd.DataFrame(data)

label_encoder = preprocessing.LabelEncoder()

enc = label_encoder.fit(df['Prediction'])
df['Prediction'] = enc.transform(df['Prediction'])  # 2. Encode outcome as they are not numerical

dta = pd.DataFrame(df,
                   columns="Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 Q11 Q12 Q13 Q14 Q15 Q16 Q17 Q18 Q19 Q20 Q21 Q22 Q23 Q24 Q25 Q26 Q27 Q28 Q29 Q30 Q31 Q32 Q33 Q34 Q35 Q36".split(
                       " "))  # 3. Divite the frame as data & outcome
print(dta)

outcome = df['Prediction']
train_x, x_test, train_y, y_test = model_selection.train_test_split(dta, outcome, test_size=0.5)

#  Linear Regression

lm =LinearRegression()
lm.fit(train_x, train_y)
# Make predictions using the testing set
y_pred1 = lm.predict(x_test)

plt.scatter(lm.predict(train_x), lm.predict(train_x) - train_y, c='b',s=10,alpha=0.2)
plt.scatter(lm.predict(x_test), lm.predict(x_test) - y_test, c='g',s=10)
plt.hlines(y=0,xmin=0,xmax=20)
plt.title('Residual Plot of LinearRegression using trainig (blue) and test (green) data')
plt.ylabel('Residual')
plt.show()
print('Linear Regression')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

# SVM

print('SVM Precision evaluation :')
svclassifier = SVC(kernel='linear')
svclassifier.fit(train_x, train_y)
y_pred2 = svclassifier.predict(x_test)

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

# KMap

scaler = StandardScaler()
scaler.fit(train_x)

classifier = KNeighborsClassifier(n_neighbors=30)
classifier.fit(train_x, train_y)
y_pred3 = classifier.predict(x_test)

y_pred3 = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred3))
print(classification_report(y_test, y_pred3))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(x_test)
    error.append(pd.np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

# AdaBoost Classification
print('Adaboost Classifier: ')
kfold = model_selection.KFold(n_splits=10, random_state=7)
model = AdaBoostClassifier(n_estimators=30, random_state=7)
results = model_selection.cross_val_score(model, train_x, train_y, cv=kfold)
print('Mean Accuracy: ', results.mean())
