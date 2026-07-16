import webbrowser
from threading import Timer

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    marks = float(request.form["marks"])

    prediction = model.predict([[attendance, marks]])[0]

    return render_template(
        "result.html",
        prediction=prediction,
        attendance=attendance,
        marks=marks
    )

@app.route("/about")
def about():
    return render_template("about.html")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)