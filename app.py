from flask import Flask, request, jsonify, render_template
import os
from flask_cors import  cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)


class ClientApp:
    def __init__(self):
        pass


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image_data = request.json['image']
        filename = "inputImage.jpg"
        decodeImage(image_data, filename)
        classifier = PredictionPipeline(filename)
        result = classifier.predict()
        print("Result before jsonify:", result)
        print("Type of result:", type(result))

        return jsonify(result)
    except KeyError:
        return "Error: 'image' key not found in request JSON"


if __name__ == '__main__':
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080)
