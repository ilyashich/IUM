import requests
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    app_url = 'http://localhost:5000'

    data = pd.read_csv("../data/data_with_categories.csv")

    train_set, test_set = train_test_split(data, test_size=0.2, random_state=21)

    test_set["session_id"] = range(1, 1+len(test_set))
    json_to_prediction = test_set.to_json(orient="records")
    requests.post(app_url + '/predict/mlp', json=json_to_prediction)
    requests.post(app_url + '/predict/naive', json=json_to_prediction)
