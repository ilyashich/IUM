from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaiveClassifier:
    def __init__(self, X, y):
        self.__dummy_model = DummyClassifier(strategy="most_frequent")
        self.__dummy_model.fit(X, y)

    def get_score(self, X, y):
        return self.__dummy_model.score(X, y)

    def dummy_classifier_predict(self, X):
        return self.__dummy_model.predict(X)


def read_data():
    data = pd.read_csv("../data/data_with_categories.csv")
    col_to_predict = "successful"
    col_user_id = "user_id"

    y = data[col_to_predict]
    X = data.drop([col_to_predict, col_user_id], axis=1)
    return X, y


if __name__ == '__main__':
    X, y = read_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    dummy_model = NaiveClassifier(X_train, y_train)
    y_pred = dummy_model.dummy_classifier_predict(X_test)
    print("Accuracy of Naive Classifier : ", accuracy_score(y_test, y_pred))
