import pickle
from flask import Flask, jsonify, request


def predict(val_x, val_y, model):
    try:

        with open(f'5073_tasca03/models/{model}.pck', 'rb') as f:
            model = pickle.load(f)

        pred = model.predict_proba([[val_x, val_y]])
        return pred

    except Exception:
        return "Error"


def create_answer(prediction, model):
    return {
        'Model':model,
        'Setosa':f"{round(prediction[0][0] * 100, 2)}%",
        'Versicolor':f"{round(prediction[0][1] * 100, 2)}%",
        'Virginica':f"{round(prediction[0][2] * 100, 2)}%"
    }

def predict_and_create_answer(val_x, val_y, model):
    prediction = predict(val_x, val_y, model)
    if "Error" in prediction:
        return "Error indicando el modelo"
    answer = create_answer(prediction, model)
    return answer

def create_all(val_x, val_y):
    result = []
    result.append(predict_and_create_answer(val_x, val_y, 'lr'))
    result.append(predict_and_create_answer(val_x, val_y, 'svm'))
    result.append(predict_and_create_answer(val_x, val_y, 'dt'))
    result.append(predict_and_create_answer(val_x, val_y, 'knn'))

    return result

app = Flask('irisPredict')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    detalls = request.get_json()
    model = detalls.get('model', 'ALL')
    if model.upper() == "ALL" or model is None:
        result = create_all(detalls['X'], detalls['Y'])
    else:
        result = predict_and_create_answer(detalls['X'], detalls['Y'], model)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
