import os
import re
from werkzeug.datastructures import FileStorage

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch

from application_code.inference_handler import BaseHandler

class PresetFilterConvolver(BaseHandler):

    @classmethod
    def most_likely_objects(cls,
                            img: FileStorage,
                            size: int = 5):

        # Step 1: parse as Pillow image
        img = Image.open(img)

        # Step 2: Transform image to match model specifications
        transformed_image = cls.transform_image_gray(img)

        # Step 3: Unsqueeze tensor
        batch_t = torch.unsqueeze(transformed_image, 0)

        # Step 4: Convolve over image
        convolution = cls.convolve_sobel_edge(batch_t)

        # back to image format
        return cls.back_to_image(convolution)

    @classmethod
    def convolve_sobel_edge(cls,
                            image):

        # define filter
        kernel_horizontal = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_vertical = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])

        # convolve filter
        image_horizontal = F.conv2d(image, kernel_horizontal)
        image_vertical = F.conv2d(image, kernel_vertical)

        # merge both
        G = (image_horizontal*image_horizontal + image_vertical*image_vertical)**0.5

        return G

    @classmethod
    def back_to_image(cls,
                      tensor):

        # Back to tensor
        squeezed_tensor = torch.squeeze(tensor)

        # back to image
        pil_image = transforms.functional.to_pil_image(squeezed_tensor)

        return pil_image





















