from flask import Flask, request, session, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

from application_code.pretrained_torchmodels import PretrainedTorchVision
from application_code.pretrained_segmentation_models import PretrainedSegmentationModels
from application_code.pretrained_detection_models import PretrainedDetectionModels
from application_code.preset_filters import PresetFilterConvolver
from toolbox.image_serving import serve_pil_image

# Set up app
app = Flask(__name__)
app.config.from_object('config_flask.DevelopmentConfig')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'TMl6zKhv'
db = SQLAlchemy(app)
cors = CORS(app, resources={r"/imagenet": {"origins": "http://localhost:3000"},
                            r"/edgedetection": {"origins": "http://localhost:3000"},
                            r"/objectdetection": {"origins": "http://localhost:3000"},
                            r"/segmentdetection": {"origins": "http://localhost:3000"}})

@app.route('/inference', methods=["GET", "POST", "OPTIONS"])
@cross_origin(origin='localhost',headers=['Content-Type'])
def imagenet():

    img = request.files['myImage']
    results = PretrainedTorchVision.most_likely_objects(img)

    return jsonify(results)

@app.route('/edgedetection', methods=["GET", "POST", "OPTIONS"])
@cross_origin(origin='localhost', headers=['Content-Type'])
def edgedetection():

    img = request.files['myImage']
    img_convolved = PresetFilterConvolver.most_likely_objects(img)

    return serve_pil_image(img_convolved)

@app.route('/objectdetection', methods=["GET", "POST", "OPTIONS"])
@cross_origin(origin='localhost', headers=['Content-Type'])
def objectsegmentation():

    img = request.files['myImage']
    img_objects = PretrainedDetectionModels.detect_objects(img)
    return serve_pil_image(img_objects)

@app.route('/segmentdetection', methods=["GET", "POST", "OPTIONS"])
@cross_origin(origin='localhost', headers=['Content-Type'])
def segmentdetection():

    img = request.files['myImage']
    img_segments = PretrainedSegmentationModels.detect_objects(img)
    return serve_pil_image(img_segments)


if __name__ == '__main__':
    app.run(host="localhost", port=7050)