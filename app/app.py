import os
from flask import Flask, render_template, request

app = Flask(__name__, template_folder = 'templates')
UPLOAD_FOLDER = 'app/static'

@app.route("/", methods = ["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            return render_template("index.html", prediction = 1)
    return render_template("index.html", prediction = 0)

if __name__ == "__main__":
    app.run(debug = True)

