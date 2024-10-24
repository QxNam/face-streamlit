import streamlit as st
import cv2, os
import numpy as np
import detect as _det
import tracking as _tra
from face_alignment import views as _v
import utils as _ut
from time import time

st.set_page_config(
    page_title="Register", 
    page_icon="💾", 
    # layout="wide"
)
st.logo('assets/bbsw_logo.png')
st.markdown("<h1 style='text-align: center; color: black;'>Register</h1>", unsafe_allow_html=True)
st.sidebar.header("Step")

st.session_state = {}
# st.session_state["DATAS"] = {data['name']: parse_faces(data['name']) for data in parse_data()}
# st.session_state["ID_REGISTER"] = len(st.session_state["DATAS"])
st.session_state["NAME_REGISTER"] = ''
st.session_state["IMAGES_REGISTER"] = {}
st.session_state["CHECK_CAM"] = False
st.session_state["IDX"] = 0

with st.form(key="form"):
    st.session_state["NAME_REGISTER"] = st.text_input("Input your name", key="NAME")
    if st.form_submit_button(label="submit", disabled=False):
        if st.session_state['NAME_REGISTER'] == '':
            st.error("Name cannot be empty.")
        elif st.session_state["NAME_REGISTER"] in ['qxnam', 'hung']: #st.session_state["DATAS"].keys()
            st.error(f"'{st.session_state['NAME_REGISTER']}' is already registered.")
        else:
            st.session_state["CHECK_CAM"] = True
            st.success(f"Submit \'{st.session_state['NAME_REGISTER']}\' successfully!")

if st.session_state["CHECK_CAM"]:
    with st.container(border=True):
        frame_counter = 0
        views = ['straight', 'left', 'right', 'up', 'down']
        start_time = time()
        st.title(views[st.session_state["IDX"]])
        FRAME_WINDOW = st.image([])
        CAMERA = cv2.VideoCapture(1)
        faces = {}
        images = {}
        
        while len(faces)<5:
            frame_counter += 1
            ret, frame = CAMERA.read()
            if ret is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps = frame_counter / (time() - start_time)
            try:
                outputs, img_info, bboxes, landmarks = _det.detector.detect_tracking(image=frame, max_num=1)
                online = _tra.tracking_frame(outputs, img_info)
                view = _v.get_view(bboxes[0], landmarks[0])
                # x, y, w, h = tuple(map(int, online[0][0]))
                # line = int((w)*0.1)
                # _ut.draw_detect(frame, (x, y), (x+w, y+h), _ut.COLORS[0], 2, 3, line)
                # _ut.put_text_rect(frame, str(view), (x, y-10), colorT=(0, 0, 0), colorR=_ut.COLORS[0])
                print(view.split(' ')[0], views[st.session_state["IDX"]])
                if view.split(' ')[0] == views[st.session_state["IDX"]]:
                    time.sleep(5)
                    face = _det.crop_after_align(frame, landmarks[0])
                    st.session_state["IMAGES_REGISTER"][view] = face
                    st.session_state["IDX"] += 1
                    st.image(face, width=400)
                if st.session_state["IDX"] == 5:
                    break
            except Exception as e:
                print(f"Error: {e}")

            FRAME_WINDOW.image(frame)
        for view in st.session_state['IMAGES_REGISTER']:
            st.image(st.session_state['IMAGES_REGISTER'][view], width=400)

        if not st.session_state["TURN_ON_CAMERA"]:
            CAMERA.release()
