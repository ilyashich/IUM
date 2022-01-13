from flask import Flask, jsonify, request
import controller

app = Flask(__name__)

ab_logic = controller.Controller()


@app.route('/predict', methods=['GET'])
def get_prediction():
    data = request.get_json(force=True)
    prediction = ab_logic.handle_predict_request(data)

    return f"{{\"prediction\": \"{prediction}\"}}"


if __name__ == '__main__':
    app.run(debug=True)
