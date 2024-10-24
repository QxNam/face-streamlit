import os
import cv2
import argparse
import toml
import torch
from utils import encode_image_to_base64, BASE
from pprint import pprint

from qdrant_db.utils import insert_data, get_views
from qdrant_db.model import Face
from qdrant_client import QdrantClient

# from face_detection.scrfd.detector import SCRFD
# from face_alignment import alignment as _al
import detect as _det
import recognize as _rec

os.makedirs('data/faces', exist_ok=True)
config = toml.load("configs.toml")['QDRANT']
# detector = SCRFD(os.path.join(BASE, 'face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx'))

def add_persson(register_dir, face_dir):
    qdrant_client = QdrantClient(
        url=config['URL'],
        api_key=config['API_KEY'],
    )
    # results = get_views(qdrant_client, 'qxnam')
    # print(results)
    # exit()
    # create_collection(qdrant_client)
    try:
        for idx, name in enumerate(os.listdir(register_dir)):
            name_path = os.path.join(register_dir, name)
            face_path = os.path.join(face_dir, name)
            os.makedirs(face_path, exist_ok=True)
            print(name)
            embeddings = {}
            faces_base64 = {}
            for view in os.listdir(name_path):
                print(f"-> {view}")
                if view.endswith(("png", "jpg", "jpeg")):
                    img = cv2.imread(os.path.join(name_path, view))
                    face = img.copy()
                    bboxes, landmarks = _det.detector.detect(image=img, max_num=1)
                    if len(landmarks) > 0:
                        x, y, xw, yh, _ = bboxes[0]
                        face = _det.crop_after_align(img, landmarks[0])
                    
                    path_save_face = os.path.join(face_path, f"{view}")
                    cv2.imwrite(path_save_face, face)

                    # embedding
                    direct = view.split('.')[0]
                    faces_base64[direct] = encode_image_to_base64(face)
                    embeddings[direct] = _rec.get_feature(face).ravel().tolist()
                    
            external_data = {
                "id": idx,
                "name": name,
                "views": faces_base64
            }
            insert_data(qdrant_client, embeddings, Face(**external_data))
    except Exception as e:
        print(f"Error occurred: {e}")
    qdrant_client.close()

def search_test():
    # qdrant_client = QdrantClient(
    #     url=config['URL'],
    #     api_key=config['API_KEY'],
    # )
    # img = cv2.imread('data/test.jpg')
    # face = img.copy()
    # _, landmarks = detector.detect(image=img, max_num=1)
    # if len(landmarks) > 0:
    #     face = _al.norm_crop(face, landmarks[0])
    #     face = detector.crop_face(face)
    
    # cv2.imwrite("data/face.jpg", face)

    face = cv2.imread('data/0_face.jpg')
    # vector = get_feature(face).tolist()
    name, score = _rec.recognize_v1(face, 0.1)
    print(name, score)
    # qdrant_client.close()

    # search by vector
    # results = search_v1(qdrant_client, vector, limit=10)
    # for hits in results:
    #     for hit in hits:
    #         print(f'{hit.payload["name"]}: {hit.score}')
    # print('-'*100)
    # results = search_v2(qdrant_client, vector)
    # for group in results.groups:
    #     for hit in group.hits:
    #         print(f'{hit.payload["name"]}: {hit.score}')
    
            # print(f'{hit.payload["name"]}: {hit.score}')
    # print('-'*100)
    # results = search_v2(qdrant_client, vector)
    # for group in results.groups:
    #     for hit in group.hits:
    #         print(f'{hit.payload["name"]}: {hit.score}')
    # qdrant_client.close()
        
if __name__ == "__main__":

    register_dir = './data/register'
    face_dir = './data/faces'
    add_persson(register_dir, face_dir)
    # search_test()

    # # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--backup-dir",
    #     type=str,
    #     default="./datasets/backup",
    #     help="Directory to save person data.",
    # )
    # parser.add_argument(
    #     "--add-persons-dir",
    #     type=str,
    #     default="./datasets/new_persons",
    #     help="Directory to add new persons.",
    # )
    # parser.add_argument(
    #     "--faces-save-dir",
    #     type=str,
    #     default="./datasets/data/",
    #     help="Directory to save faces.",
    # )
    # parser.add_argument(
    #     "--features-path",
    #     type=str,
    #     default="./datasets/face_features/feature",
    #     help="Path to save face features.",
    # )
    # opt = parser.parse_args()

    # # Run the main function
    # add_persons(**vars(opt))