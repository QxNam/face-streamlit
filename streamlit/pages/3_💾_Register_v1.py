import streamlit as st
import cv2, os
import numpy as np
from time import time
from qdrant_client import QdrantClient
from qdrant_db.model import Face
from qdrant_db.utils import insert_data, get_length, get_names
from streamlit_image_select import image_select
import detect as _det
from face_alignment import alignment as _ali
import recognize as _rec
import utils as _ut

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
st.session_state["FACE_REGISTER"] = {}
qdrant_client = QdrantClient(
    url=_ut.DATABASE['URL'],
    api_key=_ut.DATABASE['API_KEY'],
)

_name, _straight, _left, _right, _up, _down, _finish= st.tabs(['name', 'straight', 'left', 'right', 'up', 'down', 'finish'])

with _name:
    with st.form(key="form"):
        st.session_state["NAME_REGISTER"] = st.text_input("Input your name", key="NAME")
        if st.form_submit_button(label="submit", disabled=False):
            if st.session_state['NAME_REGISTER'] == '':
                st.error("Name cannot be empty.")

            # elif st.session_state["NAME_REGISTER"] in get_names(qdrant_client): #st.session_state["DATAS"].keys()
            #     st.error(f"'{st.session_state['NAME_REGISTER']}' is already registered.")
            else:
                st.success(f"Submit \'{st.session_state['NAME_REGISTER']}\' successfully!")

with _straight:
    with st.container(border=True):
        img_file_buffer = st.camera_input("Look straight", key="straight")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state["IMAGES_REGISTER"]['straight'] = image
            bboxes, landmarks = _det.detector.detect(image=image, max_num=1)
            st.session_state["FACE_REGISTER"]['straight'] = _det.crop_after_align(image, landmarks[0])
            # img = image_select(
            #     label="result",
            #     images=[
            #         st.session_state["IMAGES_REGISTER"]['straight'],
            #         st.session_state["FACE_REGISTER"]['straight']
            #     ],
            #     captions=["image", "face"],
            #     use_container_width=False
            # )
            st.image(st.session_state["FACE_REGISTER"]['straight'])

with _left:
    with st.container(border=True):
        img_file_buffer = st.camera_input("Look left", key="left")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state["IMAGES_REGISTER"]['left'] = image
            bboxes, landmarks = _det.detector.detect(image=image, max_num=1)
            st.session_state["FACE_REGISTER"]['left'] = _det.crop_after_align(image, landmarks[0])
            # img = image_select(
            #     label="result",
            #     images=[
            #         st.session_state["IMAGES_REGISTER"]['left'],
            #         st.session_state["FACE_REGISTER"]['left']
            #     ],
            #     captions=["image", "face"],
            #     use_container_width=False
            # )
            st.image(st.session_state["FACE_REGISTER"]['left'])

with _right:
    with st.container(border=True):
        img_file_buffer = st.camera_input("Look right", key="right")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state["IMAGES_REGISTER"]['right'] = image
            bboxes, landmarks = _det.detector.detect(image=image, max_num=1)
            st.session_state["FACE_REGISTER"]['right'] = _det.crop_after_align(image, landmarks[0])
            # img = image_select(
            #     label="result",
            #     images=[
            #         st.session_state["IMAGES_REGISTER"]['right'],
            #         st.session_state["FACE_REGISTER"]['right']
            #     ],
            #     captions=["image", "face"],
            #     use_container_width=False
            # )
            st.image(st.session_state["FACE_REGISTER"]['right'])

with _up:
    with st.container(border=True):
        img_file_buffer = st.camera_input("Look up", key="up")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state["IMAGES_REGISTER"]['up'] = image
            bboxes, landmarks = _det.detector.detect(image=image, max_num=1)
            st.session_state["FACE_REGISTER"]['up'] = _det.crop_after_align(image, landmarks[0])
            # img = image_select(
            #     label="result",
            #     images=[
            #         st.session_state["IMAGES_REGISTER"]['up'],
            #         st.session_state["FACE_REGISTER"]['up']
            #     ],
            #     captions=["image", "face"],
            #     use_container_width=False
            # )
            st.image(st.session_state["FACE_REGISTER"]['up'])

with _down:
    with st.container(border=True):
        img_file_buffer = st.camera_input("Look down", key="down")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state["IMAGES_REGISTER"]['down'] = image
            bboxes, landmarks = _det.detector.detect(image=image, max_num=1)
            st.session_state["FACE_REGISTER"]['down'] = _det.crop_after_align(image, landmarks[0])
            # img = image_select(
            #     label="result",
            #     images=[
            #         st.session_state["IMAGES_REGISTER"]['down'],
            #         st.session_state["FACE_REGISTER"]['down']
            #     ],
            #     captions=["image", "face"],
            #     use_container_width=False
            # )
            st.image(st.session_state["FACE_REGISTER"]['down'])

with _finish:
    # for view in st.session_state["IMAGES_REGISTER"]:
    #     image = np.array(st.session_state["IMAGES_REGISTER"][view])
    #     bboxes, landmarks = _det.detector.detect(image=image, max_num=1)
    #     face = _det.crop_after_align(image, landmarks[0])
    #     st.session_state["IMAGES_REGISTER"][view] = face
    if len(st.session_state["IMAGES_REGISTER"].keys())==5:
        img = image_select(
            label="All view",
            images=[
                np.array(st.session_state["FACE_REGISTER"]["straight"]),
                np.array(st.session_state["FACE_REGISTER"]["left"]),
                np.array(st.session_state["FACE_REGISTER"]["right"]),
                np.array(st.session_state["FACE_REGISTER"]["up"]),
                np.array(st.session_state["FACE_REGISTER"]["down"])
            ],
            captions=["look", "left", "right", "up", "down"],
            use_container_width=False
        )
    if st.button("Finish"):
        
        # get length points in qdrant_client
        idx = get_length(qdrant_client)
        embeddings = {}
        faces_base64 = {}

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        os.makedirs(f'data/faces/{st.session_state["NAME_REGISTER"]}', exist_ok=True)
        os.makedirs(f'data/register/{st.session_state["NAME_REGISTER"]}', exist_ok=True)
        for percent_complete, view in enumerate(st.session_state["FACE_REGISTER"]):
            my_bar.progress(percent_complete + 1, text=f'{view}')
            face = st.session_state["FACE_REGISTER"][view]
            face = face[:,:,::-1]
            image = st.session_state["IMAGES_REGISTER"][view]
            image = image[:,:,::-1]
            
            cv2.imwrite(f'data/register/{st.session_state["NAME_REGISTER"]}/{view}.jpg', image)
            cv2.imwrite(f'data/faces/{st.session_state["NAME_REGISTER"]}/{view}.jpg', face)
            # embedding
            # faces_base64[view] = _ut.encode_image_to_base64(face)
            # embeddings[view] = _rec.get_feature(face).ravel().tolist()
                    
            # external_data = {
            #     "id": idx,
            #     "name": st.session_state["NAME_REGISTER"],
            #     "views": faces_base64
            # }
        # insert_data(qdrant_client, embeddings, Face(**external_data))
        my_bar.empty()
        st.success(f"Register '{st.session_state['NAME_REGISTER']}' successfully!")
    qdrant_client.close()