import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp.mlp import MLP
from naive_classifier.naive_classifier import NaiveClassifier


def parse_data(json_input):
    if isinstance(json_input, str):
        parsed = json.loads(json_input)
    else:
        parsed = json.loads(json.dumps(json_input))
    data = pd.DataFrame(parsed)

    col_session_id = "session_id"
    col_success = "successful"

    session_id = data[col_session_id]
    success = data[col_success]

    return data.drop([col_session_id, col_success], axis=1), session_id, success


def create_df(session_id, prediction, success):
    df = pd.DataFrame()
    df["session_id"] = session_id
    df["prediction"] = prediction
    df["correct_value"] = success
    return df


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

    def predict_mlp(self, json_input):
        data_input, session_id, success = parse_data(json_input)

        prediction = self.__mlp.mlp_predict(data_input)

        df = create_df(session_id, prediction, success)

        json_log = df.to_json(orient="records")

        logging.info(f"Model: MLP; {json_log}")

        return {"model": "MLP", "predictions": df.to_dict(orient="records")}

    def predict_naive(self, json_input):
        data_input, session_id, success = parse_data(json_input)

        prediction = self.__dummy_model.dummy_classifier_predict(data_input)

        df = create_df(session_id, prediction, success)

        json_log = df.to_json(orient="records")

        logging.info(f"Model: Naive; {json_log}")

        return {"model": "Naive", "predictions": df.to_dict(orient="records")}
