import cv2, os
import numpy as np
import utils as _ut
from face_detection.scrfd.detector import SCRFD
from face_alignment import alignment as _ali

detector = SCRFD(os.path.join(_ut.BASE, 'face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx'))

def register_face():
    pass

################################################################

def crop_after_align(frame:np.ndarray, landmark:tuple):
    face = _ali.norm_crop(frame, landmark)
    face = detector.crop_face(image=face)
    return face

################################################################

def detect_v1(frame:np.ndarray, num_faces:int=1, alignment:bool=False):
    '''
    Detect faces in a frame.

    Args:
    frame (numpy.ndarray): Input image.
    num_faces (int): Maximum number of faces to detect.
    alignment (bool): Whether to perform face alignment before cropping.

    Returns:
        dict: Detected faces with bounding boxes, landmarks, and optionally face views.
    '''
    faces = []
    bboxes, landmarks = detector.detect(image=frame, max_num=num_faces)
    for i, bbox in enumerate(bboxes):
        x, y, xw, yh, _ = bbox
        face = frame[y:yh, x:xw]
        if alignment:
            face = crop_after_align(frame, landmarks[i])
        faces.append(face)
        line = int((xw-x)*0.1)
        _ut.draw_detect(frame, (x, y), (xw, yh), _ut.COLORS[i%5], 2, 3, line)
        _ut.put_text_rect(frame, str(i), (x, y-10), colorT=(0, 0, 0), colorR=_ut.COLORS[i%5])
    return faces

################################################################
################################################################

# from scrfd import SCRFD, Threshold
# from PIL import Image 
# face_detector = SCRFD.from_path(os.path.join(Base, 'face_detection/scrfd/weights/scrfd.onnx'))
# threshold = Threshold(probability=0.4)

# def extract_v2(frame:np.ndarray, num_faces:int=1):
#     frame = Image.fromarray(frame) 
#     faces = face_detector.detect(frame, threshold=threshold)
#     frame = np.array(frame)
#     for i, face in enumerate(faces[:num_faces]):
#         bbox = face.bbox
#         kps = face.keypoints
#         score = face.probability
#         x, y, xw, yh = int(bbox.upper_left.x), int(bbox.upper_left.y), int(bbox.lower_right.x), int(bbox.lower_right.y)
#         line = int((xw-x)*0.1)
#         draw_detect(frame, (x, y), (xw, yh), clors[i], 2, 3, line)
#     cv2.imwrite('test.jpg', frame)
    

if __name__ == "__main__":
    # pass
    img = cv2.imread('data/test.jpg')
    faces = detect_v1(img, alignment=True)
    cv2.imwrite('data/face_detect.jpg', img)
    for idx, face in enumerate(faces):
        cv2.imwrite(f'data/{idx}_face.jpg', face)