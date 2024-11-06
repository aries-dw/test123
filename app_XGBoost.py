from catboost import CatBoostClassifier, CatBoostRegressor
from flask import Flask, request, jsonify, render_template
import pickle


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("yield_XGBoost.sav", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    json_data = request.get_json()
    print(json_data)
    float_features = [#float(json_data['No']),
                      #float(json_data['Yield']),
                      float(json_data['SeedQty']),
                      float(json_data['Moisture']),
                      float(json_data['DaysCount']),
                      float(json_data['Transplanting']),
                      float(json_data['fertiliser']),
                      float(json_data['service']),
                      float(json_data['waterManagement']),
                      float(json_data['ndvi_1']),
                      float(json_data['ndvi_2']),
                      float(json_data['ndvi_3']),
                      float(json_data['hurs']),
                      float(json_data['pr']),
                      float(json_data['svcwind']),
                      float(json_data['tasmax'])]
    # Prepare features as a 2D list (expected by model)
    features = [float_features]

    predictions = model.predict(features)
    return jsonify({'predictions': predictions.tolist()})


if __name__ == "__main__":
    flask_app.run(debug=True)