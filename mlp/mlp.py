import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLP:
    def __init__(self, X, y):
        self.__mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                                   random_state=1)
        self.__mlp.fit(X, y)

    def mlp_get_score(self, X, y):
        y_pred = self.mlp_predict(X)
        return accuracy_score(y, y_pred)

    def mlp_predict(self, X):
        pred_proba = self.__mlp.predict_proba(X)
        y_pred = []
        for prediction in pred_proba:
            if prediction[1] < 0.65:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred


def read_data():
    data = pd.read_csv("../data/data_with_categories.csv")
    col_to_predict = "successful"

    y = data[col_to_predict]
    X = data.drop([col_to_predict], axis=1)
    return X, y


if __name__ == '__main__':
    X, y = read_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    mlp = MLP(X_train, y_train)

    print("Accuracy of MLPClassifier : ", mlp.mlp_get_score(X_test, y_test))
