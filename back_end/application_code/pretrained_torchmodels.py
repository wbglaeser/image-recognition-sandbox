import os
import re
from werkzeug.datastructures import FileStorage

from PIL import Image
import torchvision.models as models
import torch

from application_code.inference_handler import BaseHandler
from config import DATA_DIR

class PretrainedTorchVision(BaseHandler):

    # initialse models
    resnet = models.resnet101(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    mobilenet = models.mobilenet_v2(pretrained=True)

    # Load classification labels
    with open(os.path.join(DATA_DIR, 'imagenet_classes.txt')) as f:
        classes = [line.strip() for line in f.readlines()]

    # Regex to clean object name
    OBJECT_REGEX = re.compile('|'.join([
        r'^n[0-9]+\s'
    ]))

    @classmethod
    def inference(cls,
                  batch_t,
                  model,
                  size):

        # Set resnet into inference mode
        model.eval()

        # Produce classifcation vector
        out = model(batch_t)

        return cls.wrap_results_in_dict(out)


    @classmethod
    def wrap_results_in_dict(cls,
                             out):

        # Return most likely objects
        _, indices = torch.sort(out, descending=True)

        # Compute corresponding percentages
        percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100

        result_json = []
        for i,idx in enumerate(indices[0][:5]):

            object_name = cls.parse_object_name(cls.classes[idx])

            update = {
                "object": object_name,
                "percentage": "{:0.2f}".format(percentages[idx].item())
            }
            result_json.append(update)

        return result_json

    @classmethod
    def most_likely_objects(cls,
                            img: FileStorage,
                            size: int = 5) -> dict:

        # Step 1: parse as Pillow image
        img = Image.open(img)

        # Step 2: Transform image to match model specifications
        transformed_image = cls.transform_image(img)

        # Step 3: Turn image into 'batch'
        batch_t = torch.unsqueeze(transformed_image, 0)

        # Produce results and wrap in dict
        result_json = {}
        result_json["resnet"] = cls.inference(batch_t, cls.resnet, size)
        result_json["alexnet"] = cls.inference(batch_t, cls.alexnet, size)
        result_json["mobilenet"] = cls.inference(batch_t, cls.mobilenet, size)

        return result_json

    @classmethod
    def parse_object_name(cls,
                          name):

        return cls.OBJECT_REGEX.sub(r"", name)




















