from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import pandas as pd

titanic = pd.read_csv("../train.csv")

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
medage = titanic["Age"].median()
titanic["Age"] = titanic["Age"].fillna(medage)
medfare = titanic["Fare"].median()
titanic["Fare"] = titanic["Fare"].fillna(medfare)

# Attributes used for prediction
attributes = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

# Initialize learning model
model = LogisticRegression()

folds = KFold(titanic.shape[0], n_folds = 2, random_state = 1)

predictions = []
for train, test in folds:
    # Predictors used to train the model
    X = (titanic[attributes].iloc[train,:])
    # Target of what the model should predict
    y = titanic["Survived"].iloc[train]
    model.fit(X, y)
    test_y = model.predict(titanic[attributes].iloc[test,:])
    predictions.append(test_y)
next

