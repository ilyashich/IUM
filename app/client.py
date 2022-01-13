import requests
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    app_url = 'http://localhost:5000'

    data = pd.read_csv("../data/data_with_categories.csv")
    col_to_predict = "successful"
    col_user_id = "user_id"

    y = data[col_to_predict]
    X = data.drop([col_user_id], axis=1)
    train_set, test_set = train_test_split(X, test_size=0.2, random_state=21)
    no_of_rows = test_set.shape[0]

    # 0 for Naive, 1 for MLP
    model = 0
    session_id = 5300
    for row_index in range(0, 300):
        prediction_data = test_set.iloc[row_index]
        prediction_data["model"] = model
        prediction_data["session_id"] = session_id

        json_to_prediction = prediction_data.to_dict()

        #  send data to predict, and then send how the session actually ended
        requests.get(app_url + '/predict', json=json_to_prediction)

        if row_index % 1000 == 0:
            print(f"[{row_index:4.0f}/{no_of_rows}]")

        session_id += 1
