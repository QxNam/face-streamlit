import streamlit as st
from streamlit_image_select import image_select
from qdrant_client import QdrantClient
import utils as _ut
from qdrant_db.utils import get_length, get_payload_by_name, get_names

st.set_page_config(
    page_title="Members", 
    page_icon="🧑‍💻", 
    layout="wide"
)
st.logo('assets/bbsw_logo.png')

qdrant_client = QdrantClient(
    url=_ut.DATABASE['URL'],
    api_key=_ut.DATABASE['API_KEY'],
)

st.session_state = {}
st.session_state["DATAS"] = get_names(qdrant_client)

st.markdown("<h1 style='text-align: center; color: black;'>Members</h1>", unsafe_allow_html=True)
st.sidebar.header("Settings")

with st.sidebar:
    st.write(f"Number of members: {get_length(qdrant_client)}")
    st.session_state["NAME"] = st.selectbox("Select member", st.session_state["DATAS"])

with st.container():
    payload = get_payload_by_name(qdrant_client, st.session_state["NAME"])
    # st.session_state["MEMBER"] = st.session_state["DATAS"][st.session_state["NAME"]]
    st.write(f"ID register: {payload['id']}")
    st.write(f"Name register: {payload['name']}")
    st.write(f"Create at: {payload['register_date']}")
    views = {direct: _ut.load_image_from_base64(payload['views'][direct]) for direct in payload['views']}
    # # st.write(st.session_state["IMAGES"])
    img = image_select(
        label="All view",
        images=[
            views["straight"],
            views["left"],
            views["right"],
            views["up"],
            views["down"]
        ],
        captions=["straight", "left", "right", "up", "down"],
        use_container_width=False
    )
    # st.image(add_sunglasses(img, position))
    # st.write("hello")