from flask import Flask, render_template, request, send_from_directory

from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        customer_input_dict = {}
        customer_input_dict["age"] = int(request.form.get("age"))
        customer_input_dict["sex"] = request.form.get("sex")
        customer_input_dict["bmi"] = float(request.form.get("bmi"))
        customer_input_dict["children"] = request.form.get("children")
        customer_input_dict["smoker"] = request.form.get("smoker")
        customer_input_dict["region"] = request.form.get("region")

        prediction = PredictionPipeline(customer_input_dict).initiate_prediction()

        return render_template(
            "index.html",
            customer_prediction=f"Your insurance charges are: {round(prediction[0],2)}",
        )

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
