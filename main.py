import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Import Datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Training Data Processing
train = train[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
train_cleaned = train.dropna(axis = 0)
train_cleaned = train_cleaned.replace(['male', 'female', 'S', 'Q', 'C'], [0, 1, 1, 2,3])

male_age = train_cleaned.Age[train_cleaned['Sex'] == 0]
female_age = train_cleaned.Age[train_cleaned['Sex'] == 1]

male_median = male_age.median()
female_median = female_age.median()
mean_age = (male_median + female_median)/2

train_cleaned.Age = train_cleaned.Age.fillna(value = mean_age)

X = train_cleaned[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_cleaned[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X.shape)
#print(y.shape)

#Model training
model = DecisionTreeClassifier()
training = model.fit(X_train, y_train)
predict = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)
#print(predict)
#print(predict.shape)

#Validation Data processing
validation = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
validation.Fare.fillna(35.627188, inplace = True)
#validation_cleaned = validation.dropna(axis = 0)
validation_data = validation.replace(['male', 'female', 'S', 'Q', 'C'], [0, 1, 1, 2,3])

val_male_age = validation_data.Age[validation_data['Sex'] == 0]
val_female_age = validation_data.Age[validation_data['Sex'] == 1]

val_male_median = val_male_age.median()
val_female_median = val_female_age.median()
val_mean_age = (val_male_median + val_female_median)/2
validation_data.Age = validation_data.Age.fillna(value = val_mean_age)
print(validation_data.describe())

#Validation Data Prediction
validation_predict = model.predict(validation_data)
print(validation_predict)
print(validation_predict.shape)