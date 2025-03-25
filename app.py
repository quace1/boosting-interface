from flask import Flask, request, render_template
from catboost import CatBoostClassifier
import numpy as np

app = Flask(__name__)

model = CatBoostClassifier()
model.load_model("model/catboost_model.cbm")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        features = [
            float(request.form["feature1"]),
            float(request.form["feature2"]),
            float(request.form["feature3"]),
            float(request.form["feature4"]),
            float(request.form["feature5"])
        ]
        prediction = model.predict([features])[0]
        return f"Prediction: {prediction}"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
