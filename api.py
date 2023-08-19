import cv2
from flask import Flask
from flask_restful import Resource, Api, reqparse
from src.utils import MAT2b64, b642img
from predict_api import predict
import matplotlib.pyplot as plt


app = Flask(__name__)
api = Api(app)

class Mask(Resource):
    def post(self):
        parser = reqparse.RequestParser()  # initialize

        parser.add_argument('image', required=True)  # add arguments

        args = parser.parse_args()  # parse arguments to dictionary

        result_arr = predict(args["image"])

        if(result_arr is None):
            return None, 404

        return result_arr, 200

class ImageEncoder(Resource):
    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('image_abs_path', required=True)
        parser.add_argument('width', required=False)
        parser.add_argument('height', required=False)

        args = parser.parse_args()

        image_path = args["image_abs_path"]

        img = cv2.imread(image_path)

        if(args["width"] and args["height"]):
            img = cv2.resize(img, (int(args["width"]), int(args["height"])))

        return {"img": MAT2b64(img)}, 200

class ImageShow(Resource):
    def post(self):
        parser = reqparse.RequestParser()  # Initialize

        parser.add_argument('image', required=True)  # Add arguments

        args = parser.parse_args()  # Parse arguments to dictionary

        imgb64 = args["image"]

        img = b642img(imgb64)

        plt.imshow(img)
        plt.show()
        plt.waitforbuttonpress(0)

        return 201

api.add_resource(Mask, '/api/mask')

api.add_resource(ImageEncoder, '/api/img/encode')
api.add_resource(ImageShow, '/api/img/show')


if __name__ == '__main__':
    app.run()
