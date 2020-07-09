from io import BytesIO
import base64
from flask import jsonify

def serve_pil_image(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, 'JPEG', quality=70)
    img_str = base64.b64encode(buffered.getvalue())
    return img_str