import pickle
from flask import Flask, jsonify, request


def predict_lr(val_x, val_y):
    with open('5073_tasca03/regresion_logistica.pck', 'rb') as f:
        model = pickle.load(f)

    pred = model.predict_proba([[val_x, val_y]])
    return pred


def create_answer(prediction):
    return {
        'Setosa':f"{round(prediction[0][0] * 100, 2)}%",
        'Versicolor':f"{round(prediction[0][1] * 100, 2)}%",
        'Virginica':f"{round(prediction[0][2] * 100, 2)}%"
    }

app = Flask('irisPredict')

@app.route('/predict_lr', methods=['POST'])
def predict_lr_endpoint():
    detalls = request.get_json()

    prediction = predict_lr(detalls['X'], detalls['Y'])

    result = create_answer(prediction)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
