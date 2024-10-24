import base64, io, os, toml
import cv2, torch
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # green, red, blue, yellow, cyan
DEVICE = torch.device("mps:0" if torch.mps.is_available() else "cpu")
DATABASE = toml.load("configs.toml")['QDRANT']

def encode_image_to_base64(image:np.ndarray) -> str:
    """
    Encode an image to a base64 string.
    Args:
        image: The image to encode.
    Returns:
        str: The base64 encoded string of the image.
    """

    # Convert the image to a JPEG or PNG format
    success, encoded_image = cv2.imencode('.jpg', image)
    
    # Ensure the image was successfully encoded
    if not success:
        raise ValueError("Failed to encode the image.")

    # Encode the image to a base64 string
    base64_encoded = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    
    # Create the URI string
    uri = f"data:image/jpeg;base64,{base64_encoded}"

    return uri

def load_image_from_base64(uri: str) -> np.ndarray:
    """
    Load image from base64 string.
    Args:
        uri: a base64 string.
    Returns:
        numpy array: the loaded image.
    """

    encoded_data_parts = uri.split(",")

    if len(encoded_data_parts) < 2:
        raise ValueError("format error in base64 encoded string")

    encoded_data = encoded_data_parts[1]
    decoded_bytes = base64.b64decode(encoded_data)

    # similar to find functionality, we are just considering these extensions
    # content type is safer option than file extension
    with Image.open(io.BytesIO(decoded_bytes)) as img:
        file_type = img.format.lower()
        if file_type not in ["jpeg", "png"]:
            raise ValueError(f"input image can be jpg or png, but it is {file_type}")

    nparr = np.fromstring(decoded_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr

def draw_detect(frame, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_keypoints(frame, landmark, radius=3):
    for id, key_point in enumerate(landmark):
        cv2.circle(frame, tuple(key_point), radius, COLORS[id%5], -1)

def put_label(frame, label, top_left, color=(0, 255, 0)):
    x, y = top_left
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.6, color, 2)

def put_text_rect(frame, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
            colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
            offset=10, border=None, colorB=(0, 255, 0)):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv2.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(frame, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(frame, text, (ox, oy), font, scale, colorT, thickness)

################################################################

def updata_from_source():
    pass