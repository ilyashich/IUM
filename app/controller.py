import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp.mlp import MLP
from naive_classifier.naive_classifier import NaiveClassifier


class Controller:
    def __init__(self):
        logging.basicConfig(filename="model_compare.log", level=logging.DEBUG)

        data = pd.read_csv("../data/data_with_categories.csv")
        col_to_predict = "successful"
        col_user_id = "user_id"

        y = data[col_to_predict]
        X = data.drop([col_to_predict, col_user_id], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

        self.__dummy_model = NaiveClassifier(X_train, y_train)
        self.__mlp = MLP(X_train, y_train)

    def handle_predict_request(self, json_input):
        col_model = "model"
        col_session_id = "session_id"
        col_success = "successful"

        model = json_input[col_model]
        session_id = json_input[col_session_id]
        success = json_input[col_success]

        json_input.pop(col_model)
        json_input.pop(col_success)
        json_input.pop(col_session_id)

        converted_input = pd.DataFrame([json_input])

        if model == 0:
            prediction = self.__dummy_model.dummy_classifier_predict(converted_input)
            group = "Naive"
        else:
            prediction = self.__mlp.mlp_predict(converted_input)
            group = "MLP"

        logging.info(f"Model: {group}, {session_id}, {prediction[0]}, {success}")

        return prediction[0]
