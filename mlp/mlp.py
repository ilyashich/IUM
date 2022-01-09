import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(self, X, y):
        self.__mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                                   random_state=1)
        self.__mlp.fit(X, y)

    def mlp_get_score(self, X, y):
        return self.__mlp.score(X, y)

    def mlp_predict(self, X):
        return self.__mlp.predict(X)


if __name__ == '__main__':
    data = pd.read_csv('../data/data_with_categories.csv')

    training_set, validation_set = train_test_split(data, test_size=0.2, random_state=21)
    X_train = training_set.iloc[:, training_set.columns != 'successful'].values
    y_train = training_set.iloc[:, 3].values
    X_test = validation_set.iloc[:, validation_set.columns != 'successful'].values
    y_test = validation_set.iloc[:, 3].values

    mlp = MLP(X_train, y_train)
    y_pred = mlp.mlp_predict(X_test)

    print("Accuracy of MLPClassifier : ", round(mlp.mlp_get_score(X_test, y_test), 4))