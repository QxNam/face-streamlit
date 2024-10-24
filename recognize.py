import torch, os
import numpy as np

from torchvision import transforms
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features, compare_encodings
import utils as _ut

from qdrant_db.utils import search_v1, search_v0
from qdrant_client import QdrantClient

recognizer = iresnet_inference(
    model_name="r100",
    path=os.path.join(_ut.BASE, 'face_recognition/arcface/weights/arcface_r100.pth'),
    device=_ut.DEVICE
)

qdrant_client = QdrantClient(
    url=_ut.DATABASE['URL'],
    api_key=_ut.DATABASE['API_KEY'],
)

@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Define a series of image preprocessing steps
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Apply the defined preprocessing to the image
    face_image = face_preprocess(face_image).unsqueeze(0).to(_ut.DEVICE)

    # Use the model to obtain facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb

def recognize_v0(face, threshold=0.2):
    """
    Recognize the identity of a person using facial features.


    Args:
        face_image (numpy.ndarray): Input facial image.
        threshold (float): Threshold value for facial recognition.

    Returns:
        tuple (str, float): A tuple containing the name of the recognized person and their score.
    """
    vector = get_feature(face).ravel().tolist()
    out = search_v0(qdrant_client, vector, 1)[0]
    if out.score >= threshold:
        return out.payload['name'], out.score
    return 'Unknown', 0

def recognize_v1(face, threshold=1):
    """
    Recognize the identity of a person using facial features.


    Args:
        face_image (numpy.ndarray): Input facial image.
        threshold (float): Threshold value for facial recognition.

    Returns:
        tuple (str, float): A tuple containing the name of the recognized person and their score.
    """
    vector = get_feature(face).ravel().tolist()
    out = search_v1(qdrant_client, vector, 2) # return (name, score)
    if out[1] >= threshold:
        return out
    return 'Unknown', 0

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./data/face_features/feature")
def recognize_v2(face, threshold=1):
    query_emb = get_feature(face)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    if score >= threshold:
        return name, score
    return 'Unknown', 0