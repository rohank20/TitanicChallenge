import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

#Import Datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
pd.plotting.scatter_matrix(train, alpha=0.2)
plt.show()

#Selecting features
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
targets = ['Survived']
X = train[features]
X = X.replace(['male','female','S','Q','C'],[0, 1, 1, 2,3])#Convert categorical data to numerical data
y = train[targets]

#Training Data Preprocessing
imputer = SimpleImputer(strategy = 'median')
X_cleaned = pd.DataFrame(imputer.fit_transform(X))
X_cleaned.columns = X.columns
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

#Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predict = model.predict(X_val)
score = mean_absolute_error(y_val, predict)
print(score)

#Validation Data processing
X_test = test[features]
X_test = X_test.replace(['male', 'female', 'S', 'Q', 'C'], [0, 1, 1, 2,3])
X_test_cleaned = pd.DataFrame(imputer.fit_transform(X_test))
X_test_cleaned.columns = X_test.columns

#Validation Data Prediction
output = pd.DataFrame(model.predict(X_test_cleaned))

#Creating a CSV file
output.columns = ['Survived']
passengerID = X_test_cleaned[['PassengerId']]
passengerID['PassengerId'] = pd.to_numeric(passengerID['PassengerId'])
final = passengerID.join(output)
#print(final.dtypes)
pd.DataFrame(final).to_csv(r'C:\Users\ttroc\anaconda3\envs\Kaggle_Titanic\output.csv', index = False)