import sklearn as sk
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def read_data():
    data = pd.read_csv('../data/data_with_categories.csv')
    col_to_predict = 'successful'

    y = data[col_to_predict]
    X = data.drop([col_to_predict], axis=1)
    return X, y


def accuracy(conf_matrix):
    diagonal_sum = conf_matrix.trace()
    sum_of_all_elements = conf_matrix.sum()
    return diagonal_sum / sum_of_all_elements


if __name__ == '__main__':
    data = pd.read_csv('../data/data_with_categories.csv')

    training_set, validation_set = train_test_split(data, test_size=0.2, random_state=21)
    X_train = training_set.iloc[:, training_set.columns != 'successful'].values
    y_train = training_set.iloc[:, 3].values
    X_test = validation_set.iloc[:, validation_set.columns != 'successful'].values
    y_test = validation_set.iloc[:, 3].values

    classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                               random_state=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    print("Accuracy of MLPClassifier : ", accuracy(cm))
    # NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # NN.fit(X, y)
    # NN.predict(X.iloc[460:, :])
    # print(round(NN.score(X, y), 4))
