import os
import numpy as np
import cv2, yaml
from time import time
from face_tracking.tracker.byte_tracker import BYTETracker

import detect as _det
import utils as _ut
from face_alignment import views as _v

def _load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

file_name = os.path.join(_ut.BASE, "./face_tracking/config/config_tracking.yaml")
config_tracking = _load_config(file_name)
tracker = BYTETracker(args=config_tracking, frame_rate=30)

clors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
# def tracking_frame(frame):
    # outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
def tracking_frame(outputs, img_info):
    online_tlwhs = []
    online_ids = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > config_tracking["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > config_tracking["min_box_area"] and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
    return (
        online_tlwhs,
        online_ids
    )

# Function for performing object detection and tracking
def inference():
    cap = cv2.VideoCapture(1)
    start = time()
    frame_count = 0
    fps = -1
    while True:
        # Read a frame from the video capture
        ret_val, frame = cap.read()

        if ret_val:
            # online = tracking_frame(frame)
            # frame = plot_tracking(frame, online[0], online[1])
            
            outputs, img_info, bboxes, landmarks = _det.detector.detect_tracking(image=frame)
            online = tracking_frame(outputs, img_info)
            ################
            for i, tlwh in enumerate(online[0]):
                x, y, w, h = tuple(map(int, tlwh))
                label = int(online[1][i])
                line = int((w)*0.1)
                _ut.draw_detect(frame, (x, y), (x+w, y+h), clors[i%7], 2, 3, line)
                _ut.put_label(frame, str(label), (x, y-5), color=clors[i%7])
            ################

            frame_count += 1
            if frame_count >= 30:
                end = time()
                fps = frame_count / (end - start)
                frame_count = 0
                start = time()
            if fps > 0:
                cv2.putText(frame, f"FPS    : {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Members: {len(online[1])}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Face Tracking", frame)
            # Check for user exit input
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

if __name__ == "__main__":
    inference()
