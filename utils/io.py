import base64
from pathlib import Path

def encode_image(img_addr):

    with open(img_addr, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)