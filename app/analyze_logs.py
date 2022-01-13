log_file = "model_compare.log"


def read_data_from_log_file():
    file = open(log_file, "r")

    session_predictions = {}

    prediction_prefix = 'INFO:root:Model:'

    len_of_prediction_prefix = len(prediction_prefix)

    log_line = file.readline()
    while log_line:
        log_line = log_line.replace(" ", "").replace("\n", "")

        if log_line.startswith(prediction_prefix):

            log_line = log_line[len_of_prediction_prefix:]
            model, session_id, prediction, correct_result = log_line.split(',')
            session_id = str(int(float(session_id)))
            correct_result = str(int(float(correct_result)))
            session_predictions[session_id] = [model, prediction, correct_result]

        log_line = file.readline()
    return session_predictions


def count_successful_predictions(predictions):
    sum_of_correct_for_naive = 0
    sum_of_correct_for_mlp = 0
    sum_of_all_naive = 0
    sum_of_all_mlp = 0
    no_of_false_possitive_for_our_model = 0

    session_ids = predictions.keys()
    for ses_id in session_ids:
        model = predictions[ses_id][0]
        is_correct = predictions[ses_id][1] == predictions[ses_id][2]

        if model == "Naive":
            sum_of_all_naive += 1
            if is_correct:
                sum_of_correct_for_naive += 1
        elif model == "MLP":
            sum_of_all_mlp += 1
            if is_correct:
                sum_of_correct_for_mlp += 1
            if predictions[ses_id][1] == 1 and predictions[ses_id][2] == 0:
                no_of_false_possitive_for_our_model += 1

    proc_of_naive = round((sum_of_correct_for_naive/sum_of_all_naive) * 100, 2)
    proc_of_mlp = round((sum_of_correct_for_mlp/sum_of_all_mlp) * 100, 2)
    return (proc_of_naive, sum_of_all_naive), (proc_of_mlp, sum_of_all_mlp, no_of_false_possitive_for_our_model)


if __name__ == "__main__":
    predictions = read_data_from_log_file()
    (proc_of_A, sum_of_all_A), (proc_of_B, sum_of_all_B, false_positives) = count_successful_predictions(predictions)

    print(f"""Naive: skuteczność={proc_of_A:.2f}% dla {sum_of_all_A} rekordów""")
    print(f"""MLP: skuteczność={proc_of_B:.2f}% dla {sum_of_all_B} rekordów
    z czego liczba błędów false-positive={false_positives}""")