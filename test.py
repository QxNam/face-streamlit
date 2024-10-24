import streamlit as st
import cv2
import numpy as np
import threading

VIDEO_SOURCE = 1  # an integer number for an OpenCV supported camera or any video file
st.session_state['count'] = 0
class CameraThread(threading.Thread):
    def __init__(self, name='CameraThread'):
        super().__init__(name=name, daemon=True)
        self.stop_event = False
        self.open_camera()
        self._frame, self.results = np.zeros((300,300,3), dtype=np.uint8), [] #initial empty frame
        self.lock = threading.Lock()
        self.log_counter = 0

    def open_camera(self):
        self.webcam = cv2.VideoCapture(VIDEO_SOURCE)

    def run(self):
        while not self.stop_event:
            
            ret, img = self.webcam.read()
            if not ret: #re-open camera if read fails. Useful for looping test videos
                self.open_camera()

                st.session_state['count'] += 1
                continue
            # results = self.process_frame(img)
            with self.lock: self.results, self._frame = img, img.copy()

    def stop(self):
        self.stop_event = True

    def read(self):
        with self.lock: 
            return self._frame.copy(), self.results

def get_or_create_camera_thread():
    for th in threading.enumerate():
        if th.name == 'CameraThread':
            th.stop()
            th.join()
    cw = CameraThread('CameraThread')
    cw.start()
    return cw

camera = get_or_create_camera_thread()
# label_map = load_labelmap()

st_frame = st.empty()
confidence_thresh = st.sidebar.slider('Confidence Threshold', value=0.5, key='threshold')
# interested_classes = [label_map.index(l) for l in CLASSES_OF_INTEREST]


c1, c2 =  st.columns([2, 1])

with c1:
    st.write(st.session_state['count'])

with c2:
    st.session_state['count'] = 0
    while True:
        frame, detections = camera.read()
        # img = draw_boxes(frame, detections, confidence_thresh, label_map, interested_classes)
        st_frame.image(frame, channels='BGR')

