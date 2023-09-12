import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="tamplates", static_folder="staticFiles")
model = pickle.load(open(r"./build.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [[int(x) for x in request.form.values()]]
    final_features = np.array(int_features)
    prediction = model.predict(final_features)
    print(final_features)
    print(prediction)
    return render_template(
        "index.html",
        prediction_text=f"{str(np.round(np.exp(prediction),2))}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
