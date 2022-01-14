from flask import Flask, jsonify, request
import controller

app = Flask(__name__)

controller = controller.Controller()


@app.route('/predict/mlp', methods=['POST'])
def predict_mlp():
    data = request.get_json(force=True)
    prediction = controller.predict_mlp(data)

    return f"{{\"prediction\": \"{prediction}\"}}"


@app.route('/predict/naive', methods=['POST'])
def predict_naive():
    data = request.get_json(force=True)
    prediction = controller.predict_naive(data)

    return f"{{\"prediction\": \"{prediction}\"}}"


if __name__ == '__main__':
    app.run()
