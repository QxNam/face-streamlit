import streamlit as st
import cv2, os
import numpy as np
import threading
from time import time
from datetime import datetime as _dt
import detect as _det
import recognize as _rec
import tracking as _tra
from face_alignment import views as _v
from face_alignment import alignment as _ali
import utils as _ut

st.set_page_config(
    page_title="Checkin", 
    page_icon="👤", 
    layout="wide"
)
st.logo('assets/bbsw_logo.png')
st.markdown("<h1 style='text-align: center; color: black;'>Checkin</h1>", unsafe_allow_html=True)
st.sidebar.header("Settings")

st.session_state = {}
st.session_state["TURN_ON_CAMERA"] = False
st.session_state["FIRST_ENABLE_CAMERA"] = False
st.session_state["TURN_ON_DETECT"] = False
st.session_state["TURN_ON_RECOGNIZE"] = False
st.session_state["RECORDING"] = False
st.session_state["TURN_ON_RECORD"] = False
# st.session_state["VECTORS"] = []
st.session_state["ID_FACE_MAPPING"] = {}
st.session_state["DATA"] = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}

################################################## SIDEBAR ##################################################

with st.sidebar:
    if st.toggle("Camera"):
        st.session_state["FIRST_ENABLE_CAMERA"] = True
        st.session_state["TURN_ON_CAMERA"] = not st.session_state["TURN_ON_CAMERA"] # nếu bật cam thì bật cam
        # st.session_state["_RECORD"] = False # nếu bật cam thì trạng thái ghi tạm dừng
        st.session_state["RECORD"] = "pause"
    
    if st.toggle("Record", disabled=True if not st.session_state["TURN_ON_CAMERA"] else False):
        st.session_state["TURN_ON_RECORD"] = not st.session_state["TURN_ON_RECORD"] # nếu bật record thì bật ghi
        # if st.session_state["TURN_ON_RECORD"]:
        #     st.session_state["RECORDING"] = False
        
    if st.toggle("Detect", disabled=True if not st.session_state["TURN_ON_CAMERA"] else False, value=False) and st.session_state["TURN_ON_CAMERA"]:
        st.session_state["TURN_ON_DETECT"] = not st.session_state["TURN_ON_DETECT"]
        st.session_state["TURN_ON_RECOGNIZE"] = False
    
    if st.toggle("Recognize", disabled=True if not st.session_state["TURN_ON_CAMERA"] else False, value=False) and st.session_state["TURN_ON_CAMERA"]:
        st.session_state["TURN_ON_RECOGNIZE"] = not st.session_state["TURN_ON_RECOGNIZE"]
        st.session_state["TURN_ON_DETECT"] = False

    st.session_state["NUM_FACES"] = int(st.slider("Num faces", 1, 5, 1))

################################################## CONTAINER ##################################################

def mapping_bbox(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU score.
    """
    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

# def recognition(face_image) -> str:
#     """
#     Recognize a face image.

#     Args:
#         face_image: The input face image.

#     Returns:
#         tuple: A tuple containing the recognition score and name.
#     """
#     # Get feature from face
#     vector = _rec.get_feature(face_image)
#     results = search_v1(_rec.qdrant_client, vector, 1)
#     label = 'Unknown'
#     if results[1] >= 1:
#         label = f'{results[0]} - {results[1]:.2f}'
#     return label

def recognize_thread():
    while True:
        raw_image = st.session_state["DATA"]["raw_image"]
        detection_landmarks = st.session_state["DATA"]["detection_landmarks"]
        detection_bboxes = st.session_state["DATA"]["detection_bboxes"]
        tracking_ids = st.session_state["DATA"]["tracking_ids"]
        tracking_bboxes = st.session_state["DATA"]["tracking_bboxes"]
        print('>>> tracking_bboxes:', tracking_bboxes)
        if tracking_bboxes == []:
            continue
        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    face_alignment = _ali.norm_crop(img=raw_image, landmark=detection_landmarks[j])
                    st.session_state["ID_FACE_MAPPING"][tracking_ids[i]] = recognition(face_image=face_alignment)
                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)
                    break

def process_tracking(frame):
    outputs, img_info, bboxes, landmarks = _det.detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = _tra.tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            vertical = tlwh[2] / tlwh[3] > _tra.config_tracking["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > _tra.config_tracking["min_box_area"] and not vertical:
                tracking_bboxes.append(tlwh)
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(t.track_id)
                tracking_scores.append(t.score)

        # label
        for i, tlwh in enumerate(tracking_bboxes):
            x, y, w, h = tuple(map(int, tlwh))
            label = 'None'
            if tracking_ids[i] in st.session_state["ID_FACE_MAPPING"]:
                label = st.session_state["ID_FACE_MAPPING"][tracking_ids[i]]
            line = int((w)*0.1)
            _ut.draw_detect(frame, (x, y), (x+w, y+h), _tra.clors[i%5], 2, 3, line)
            _ut.put_label(frame, str(label), (x, y-5), color=_tra.clors[i%5])
    else:
        frame = img_info["raw_img"]

    st.session_state["DATA"]["raw_image"] = img_info["raw_img"]
    st.session_state["DATA"]["detection_bboxes"] = bboxes
    st.session_state["DATA"]["detection_landmarks"] = landmarks
    st.session_state["DATA"]["tracking_ids"] = tracking_ids
    st.session_state["DATA"]["tracking_bboxes"] = tracking_bboxes

    return frame

def tracking_demo(frame):
    online = _tra.tracking_frame(frame)
    # frame = plot_tracking(frame, online[0], online[1])

    ################
    for i, tlwh in enumerate(online[0]):
        x, y, w, h = tuple(map(int, tlwh))
        label = int(online[1][i])
        line = int((w)*0.1)
        _ut.draw_detect(frame, (x, y), (x+w, y+h), _tra.clors[i%5], 2, 3, line)
        _ut.put_label(frame, str(label), (x, y-5), color=_tra.clors[i%5])
    ################

with st.container(border=True):
    FRAME_WINDOW = st.image([])

    if st.session_state["TURN_ON_CAMERA"]:
        CAMERA = cv2.VideoCapture(1)
        frame_width = int(CAMERA.get(3))
        frame_height = int(CAMERA.get(4))
        size = (frame_width, frame_height)
        if st.session_state["TURN_ON_RECORD"] and not st.session_state["RECORDING"]:
            os.makedirs("videos", exist_ok=True)
            VIDEO = cv2.VideoWriter(
                f"videos/{_dt.now()}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 28, size
            )
            st.session_state["RECORDING"] = True
    else:
        st.image("assets/face-20-10.jpg", use_column_width=True) # , use_column_width=True, width=int(st.session_state['SCREEN_SHAPE'][0]*0.5)

    # if st.session_state["TURN_ON_RECOGNIZE"]:
    #     print('start recognition')
    #     t = threading.Thread(target=recognize_thread, deamon=True)
    #     print(f'live: {t.is_alive()}')
    #     t.start()
    #     print('ok')

    # else:
    #     if t1.is_alive():
    #         print('stop recognition')
    #         t1.join()
            # kill t1
    # tracker = DeepSort(max_age=30, embedder='mobilenet')
    start = time()
    frame_count = 0
    delay = 0
    fps = -1
    while st.session_state["TURN_ON_CAMERA"]:
        ret, frame = CAMERA.read()
        if ret is False:
            break
        
        ##################################################
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if st.session_state["TURN_ON_DETECT"]:
            try:
                ##--- detect without tracking
                # bboxes, landmarks = _det.detector.detect(image=frame, max_num=st.session_state["NUM_FACES"])
                # for i, bbox in enumerate(bboxes):
                #     x, y, xw, yh, _ = bbox
                #     # if show view
                #     view = _v.get_view(bbox, landmarks[i])
                #     line = int((xw-x)*0.1)
                #     _ut.draw_detect(frame, (x, y), (xw, yh), _ut.COLORS[i%5], 2, 3, line)
                #     _ut.put_text_rect(frame, str(view), (x, y-10), colorT=(0, 0, 0), colorR=_ut.COLORS[i%5])

                ##--- detect with tracking
                outputs, img_info, bboxes, landmarks = _det.detector.detect_tracking(image=frame, max_num=st.session_state["NUM_FACES"])
                online = _tra.tracking_frame(outputs, img_info)
                for i, tlwh in enumerate(online[0]):
                    x, y, w, h = tuple(map(int, tlwh))
                    label = str(online[1][i])
                    # if show view
                    view = _v.get_view(bboxes[i], landmarks[i])
                    line = int((w)*0.1)
                    _ut.draw_detect(frame, (x, y), (x+w, y+h), _ut.COLORS[i%5], 2, 3, line)
                    # _ut.draw_keypoints(frame, landmarks[i])
                    _ut.put_text_rect(frame, str(view), (x, y-10), colorT=(0, 0, 0), colorR=_ut.COLORS[i%5])

            except Exception as e:
                print(f'Error in detect: {e}')

        elif st.session_state["TURN_ON_RECOGNIZE"]:
            try:
                ## detect with build tracking
                # outputs, img_info, bboxes, landmarks = _det.detector.detect_tracking(image=frame, max_num=st.session_state["NUM_FACES"])
                # online = _tra.tracking_frame(outputs, img_info)
                # for i, tlwh in enumerate(online[0]):
                #     x, y, w, h = tuple(map(int, tlwh))
                #     # face = frame[y:y+h, x:x+w]
                #     face = _det.crop_after_align(frame, landmarks[i])
                #     name, score = _rec.recognize_v1(face=face, threshold=1)
                #     label = f'{name} - {score:.2f}'
                #     line = int((w)*0.1)
                #     _ut.draw_detect(frame, (x, y), (x+w, y+h), _ut.COLORS[i%5], 2, 3, line)
                #     _ut.put_label(frame, str(label), (x, y-5), color=_ut.COLORS[i%5])
                
                ## detect without tracking
                bboxes, landmarks = _det.detector.detect(image=frame, max_num=st.session_state["NUM_FACES"])
                for i, bbox in enumerate(bboxes):
                    x, y, xw, yh, _ = bbox
                    face = _det.crop_after_align(frame, landmarks[i])
                    name, score = _rec.recognize_v2(face=face, threshold=0.5)
                    line = int((xw-x)*0.1)
                    _ut.draw_detect(frame, (x, y), (xw, yh), _ut.COLORS[i%5], 2, 3, line)
                    _ut.put_text_rect(frame, f'{name} - {score:.2f}', (x, y), colorT=(0, 0, 0), colorR=_ut.COLORS[i%5])

            except Exception as e:
                print(f'Error in detect: {e}')
        ##################################################

        # fps
        frame_count += 1
        if frame_count >= 30:
            end = time()
            fps = frame_count / (end - start)
            frame_count = 0
            start = time()
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)

        # record 
        if st.session_state["RECORDING"]:
            if st.session_state["TURN_ON_RECORD"]:
                VIDEO.write(frame[:,:,::-1])
                delay += 1
                if delay >20 and delay < 40:
                    cv2.putText(frame, f"Rec", (frame_width-80, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
                elif delay >= 40:
                    delay = 0
            else:
                st.session_state["RECORDING"] = False
                VIDEO.release()
        
        # display
        FRAME_WINDOW.image(frame)
    
    if not st.session_state["TURN_ON_CAMERA"] and st.session_state["FIRST_ENABLE_CAMERA"]:
        CAMERA.release()
    