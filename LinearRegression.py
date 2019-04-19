import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("data\student-mat.csv", sep=";")

data = data[
    ["G1", "G2", "G3", "studytime", "absences", "failures", "traveltime", "freetime"]]
print(data.head())

predict = "G3"

# features
X = np.array(data.drop([predict], 1))
# labels to predict
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

"""
best_score = 0;
for _ in range(3500):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best_score:
        best_score = accuracy;
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

pickle_input = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_input)

# print("Accuracy: \n", best_score)
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

print("\n prediction: \t actual data: \t \t correct grade:")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

style.use("ggplot")
p = "G2"
plt.scatter(data[p], data[predict])
plt.xlabel(p)
plt.ylabel("Final grade")
plt.show()
