from toolbox.pytorch_objectdetecttrack.utils import utils, datasets, parse_config
from toolbox.pytorch_objectdetecttrack.models import *
from werkzeug.datastructures import FileStorage

import random, os
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont, ImageColor

from config import YOLO_DIR

config_path = YOLO_DIR + '/config/yolov3.cfg'
weights_path = YOLO_DIR + '/yolov3.weights'
class_path = YOLO_DIR + '/config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
colors = [ImageColor.getrgb(c.lower()) for n, c in ImageColor.colormap.items() if sum(ImageColor.getrgb(c)) < 450]

from application_code.inference_handler import BaseHandler

class PretrainedDetectionModels(BaseHandler):

    # initialse models
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)

    # Set resnet into inference mode
    model.eval()
    classes = utils.load_classes(class_path)
    Tensor = torch.FloatTensor

    @classmethod
    def detect_objects(cls,
                       img: FileStorage):

        # Step 1: parse as Pillow image
        img = Image.open(img)

        # Produce results and wrap in dict
        detections = cls.run_inference(img)
        print(detections)
        img_with_box = cls.draw_image(img, detections)

        return img_with_box

    @classmethod
    def run_inference(cls,
                      img):

        # scale and pad image
        ratio = min(img_size / img.size[0], img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)

        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
             transforms.Pad((max(int((imh - imw) / 2), 0),
                             max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2), 0),
                             max(int((imw - imh) / 2), 0)), (128, 128, 128)),
             transforms.ToTensor(),
             ])

        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        print(image_tensor.shape)
        input_img = Variable(image_tensor.type(cls.Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = cls.model(input_img)
            detections = utils.non_max_suppression(detections, 80,
                                                   conf_thres, nms_thres)
        return detections[0]

    @classmethod
    def draw_image(cls, source_img, detections):

        draw = ImageDraw.Draw(source_img, "RGBA")
        img = np.array(source_img)

        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x


        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                draw.rectangle(((x1, y1), (box_w+x1, box_h+y1)), outline=colors[int(cls_pred)], width=5)
                draw.rectangle(((x1, y1), (x1+box_w*0.4), y1+int(max(box_h*0.08, 20))), fill=colors[int(cls_pred)])

                font = ImageFont.truetype("/System/Library/Fonts/SFNSText.ttf", max(box_h*0.05, 20))
                draw.text((x1, y1), cls.classes[int(cls_pred)], "white", font)

        return source_img














