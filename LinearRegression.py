import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

data = pd.read_csv("data\student-mat.csv", sep=";")

data = data[
    ["G1", "G2", "G3", "studytime", "absences", "failures", "age", "traveltime", "studytime", "freetime", "health"]]
print(data.head())

predict = "G3"

# features
X = np.array(data.drop([predict], 1))
# labels to predict
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

print("\n prediction: \t actual data: \t \t correct grade:")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
