from werkzeug.datastructures import FileStorage
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from torchvision import models
from torchvision import transforms
import torch
import numpy as np

from application_code.inference_handler import BaseHandler

class PretrainedSegmentationModels(BaseHandler):

    # initialse models
    dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

    @classmethod
    def inference(cls,
                  batch_t,
                  model):

        # Produce classifcation vector
        out = model(batch_t)

        return out["out"]


    @classmethod
    def detect_objects(cls,
                       img: FileStorage) -> dict:

        # Step 1: parse as Pillow image
        img = Image.open(img)

        # Step 2: Transform image to match model specifications
        transformed_image = cls.transform_image(img)

        # Step 3: Turn image into 'batch'
        batch_t = torch.unsqueeze(transformed_image, 0)

        # Produce results and wrap in dict
        result = cls.inference(batch_t, cls.dlab)

        # do some numpy stuff
        om = torch.argmax(result.squeeze(), dim=0).detach().cpu().numpy()

        # translate back to color
        img = cls.decode_segmap(om)

        return img

    # Define the helper function
    @classmethod
    def decode_segmap(cls, image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)

        return Image.fromarray(rgb, 'RGB')














