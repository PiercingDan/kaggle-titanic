import pandas as pd

titanic = pd.read_csv("../train.csv")

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
medage = titanic["Age"].median()
titanic["Age"] = titanic["Age"].fillna(medage)
medfare = titanic["Fare"].median()
titanic["Fare"] = titanic["Fare"].fillna(medfare)
# print(titanic.head(5))
