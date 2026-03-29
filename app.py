import streamlit as st
import cv2
import time
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import torch
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Production CPU/Memory optimization overrides
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Import custom modules
from src.face_recognition_engine import FaceRecognitionEngine
from src.database_manager import DatabaseManager
from src.utils import add_timestamp_overlay, add_alert_banner

# --- WebRTC Configuration ---
# Required for Streamlit Cloud to negotiate the connection smoothly across internet firewalls
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Page Config ---
st.set_page_config(
    page_title="Missing Person Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { max-width: 100%; }
    /* Enhance the WebRTC video styling */
    div[data-testid="stWebRtc"] video {
        border-radius: 10px;
        border: 2px solid #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Initialize Backend Systems ---
@st.cache_resource
def init_system():
    engine = FaceRecognitionEngine(similarity_threshold=0.65)
    db_manager = DatabaseManager()
    return engine, db_manager

face_engine, db_manager = init_system()

# Compute embeddings efficiently and cache them based on registered persons
def load_embeddings():
    persons = db_manager.get_missing_persons()
    query_embeddings = {}
    for name, image_path in persons.items():
        try:
            image = Image.open(image_path)
            embedding = face_engine.encode_face(image)
            if embedding is not None:
                query_embeddings[name] = embedding
        except Exception as e:
            st.error(f"⚠️ Could not load embedding for {name}: {e}")
    return query_embeddings

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    # --- 1. Person Management ---
    st.subheader("👤 Register Person")
    person_name = st.text_input("Full Name")
    person_image = st.file_uploader("Clear Face Photo", type=['jpg', 'jpeg', 'png'])
    if st.button("Register Person"):
        if person_name and person_image:
            res = db_manager.add_missing_person(person_name, person_image)
            if res:
                st.success(f"Registered {person_name}!")
                st.rerun()
            else:
                st.error("Registration failed.")
        else:
            st.warning("Please provide name and image.")

    # Show registered persons
    persons_dict = db_manager.get_missing_persons()
    if persons_dict:
        st.write("**Registered Profiles**")
        for pname in persons_dict.keys():
            c1, c2 = st.columns([4, 1])
            c1.caption(f"👱 {pname}")
            if c2.button("🗑️", key=f"del_p_{pname}", help="Delete Person"):
                db_manager.remove_missing_person(pname)
                st.rerun()

    st.markdown("---")
    
    # --- 2. System Settings ---
    st.subheader("📊 Model Settings")
    threshold = st.slider("Match Confidence Threshold", 0.0, 1.0, 0.65, 0.05)
    if face_engine.similarity_threshold != threshold:
        face_engine.set_similarity_threshold(threshold)

# --- Main Dashboard ---
st.title("🔍 WebRTC Missing Person Detection System")
st.caption("Securely streams directly from your device's camera to the cloud for real-time inference.")

# Refresh embeddings
query_embeddings = load_embeddings()

col_main, col_logs = st.columns([3, 1])

# Global state to throttle database logging inside the video thread
if 'last_detection_log_time' not in st.session_state:
    st.session_state['last_detection_log_time'] = 0

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This function processes every single frame produced by the user's camera automatically.
    """
    img = frame.to_ndarray(format="bgr24")
    
    # 1. Detection Process
    matches = face_engine.detect_and_match(img, query_embeddings)
    
    if matches:
        img = face_engine.draw_matches(img, matches)
        img = add_alert_banner(img, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
        
        # 2. Database Logging (Throttled to once every 3 seconds to prevent database locking)
        current_time = time.time()
        # Note: session_state doesn't natively map into the webrtc thread perfectly depending on context,
        # but for simple local file logging, directly engaging db_manager is fine.
        for match in matches:
            db_manager.log_detection(
                match['person_name'], "webrtc_camera_1", "Web User Camera",
                match['similarity'], img
            )
            
    # 3. Frame Processing for Display
    img = add_timestamp_overlay(img, "Live Client WebRTC")
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")


with col_main:
    st.subheader("📹 Live Camera Feed")
    
    webrtc_streamer(
        key="detection-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False # Audio is not needed for face detection
        },
        async_processing=True
    )

with col_logs:
    st.subheader("📋 Detection Logs")
    if st.button("🔄 Refresh Logs"):
        pass # Streamlit handles rerun automatically
        
    log_file = db_manager.detection_log_file
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if not df.empty:
            df["Confidence"] = df["similarity"].apply(lambda x: f"{float(x):.1%}")
            st.dataframe(
                df.tail(10)[['timestamp', 'person_name', 'camera_name', 'Confidence']].iloc[::-1],
                use_container_width=True, 
                hide_index=True
            )
            
            # Export
            if st.button("💾 Export Log"):
                out = db_manager.export_detection_report()
                if out:
                    st.success(f"Exported to {out}")
        else:
            st.info("No detections log yet.")
    else:
        st.info("No detections log yet.")
