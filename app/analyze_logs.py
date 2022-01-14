import json

import pandas as pd

log_file = "model_compare.log"


def read_data_from_log_file():
    file = open(log_file, "r")

    mlp_predictions = pd.DataFrame()
    naive_predictions = pd.DataFrame()

    prediction_mlp_prefix = "INFO:root:Model:MLP;"
    prediction_naive_prefix = "INFO:root:Model:Naive;"

    len_of_prediction_mlp_prefix = len(prediction_mlp_prefix)
    len_of_prediction_naive_prefix = len(prediction_naive_prefix)

    log_line = file.readline()
    while log_line:
        log_line = log_line.replace(" ", "").replace("\n", "")

        if log_line.startswith(prediction_mlp_prefix):
            log_line = log_line[len_of_prediction_mlp_prefix:]
            session_id, prediction, correct_result = log_line.split(';')
            session_id = json.loads(session_id)
            prediction = json.loads(prediction)
            correct_result = json.loads(correct_result)
            df_mlp = pd.DataFrame()
            df_mlp["session_id"] = session_id
            df_mlp["prediction"] = prediction
            df_mlp["correct_result"] = correct_result
            mlp_predictions = mlp_predictions.append(df_mlp, ignore_index=True)

        if log_line.startswith(prediction_naive_prefix):
            log_line = log_line[len_of_prediction_naive_prefix:]
            session_id, prediction, correct_result = log_line.split(';')
            session_id = json.loads(session_id)
            prediction = json.loads(prediction)
            correct_result = json.loads(correct_result)
            df_naive = pd.DataFrame()
            df_naive["session_id"] = session_id
            df_naive["prediction"] = prediction
            df_naive["correct_result"] = correct_result
            naive_predictions = naive_predictions.append(df_naive, ignore_index=True)

        log_line = file.readline()
    return mlp_predictions, naive_predictions


def count_successful_predictions(predictions):
    sum_of_correct = 0
    sum_of_all = 0
    no_of_false_positive = 0

    predictions_num = predictions.shape[0]
    for number in range(0, predictions_num):
        prediction = predictions.iloc[number]
        is_correct = prediction["prediction"] == prediction["correct_result"]

        sum_of_all += 1
        if is_correct:
            sum_of_correct += 1
        if prediction["prediction"] == 1 and prediction["correct_result"] == 0:
            no_of_false_positive += 1
    if sum_of_all != 0:
        proc = round((sum_of_correct / sum_of_all) * 100, 2)
    else:
        proc = 0
    return proc, sum_of_all, no_of_false_positive


if __name__ == "__main__":
    mlp_predict, naive_predict = read_data_from_log_file()

    percent_mlp, sum_all_mlp, num_false_positive_mlp = count_successful_predictions(mlp_predict)
    percent_naive, sum_all_naive, num_false_positive_naive = count_successful_predictions(naive_predict)

    print(f"Naive: accuracy: {percent_naive:.2f}% for {sum_all_naive} predictions, number of false positives: {num_false_positive_naive}")
    print(f"MLP: accuracy: {percent_mlp:.2f}% for {sum_all_mlp} predictions, number of false positives: {num_false_positive_mlp}")
