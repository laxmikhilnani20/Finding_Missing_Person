import streamlit as st
import cv2
import time
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import torch
import tempfile
import numpy as np

# Production CPU/Memory optimization overrides
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Import custom modules
from src.face_recognition_engine import FaceRecognitionEngine
from src.ip_camera_manager import IPCameraManager
from src.database_manager import DatabaseManager
from src.utils import add_timestamp_overlay, add_alert_banner, validate_ip_url


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
    .stApp {
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Initialize Backend Systems ---
@st.cache_resource
def init_system():
    # Cache the engine and managers so they persist
    engine = FaceRecognitionEngine(similarity_threshold=0.65)
    # cam_manager = IPCameraManager()  # Removed for cloud demo
    db_manager = DatabaseManager()
    return engine, db_manager
    

face_engine, db_manager = init_system()

# Compute embeddings efficiently
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

query_embeddings = load_embeddings()

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


mode = st.radio("Select Input Mode", ["Image Upload", "Video Upload"])

if mode == "Image Upload":
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        file_bytes = uploaded_image.read()
        import numpy as np
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        if st.button("Run Detection"):
            if not query_embeddings:
                st.warning("No registered persons to detect.")
            else:
                matches = face_engine.detect_and_match(frame, query_embeddings)
                if matches:
                    st.success(f"Detect matches: {len(matches)}")
                    frame = face_engine.draw_matches(frame, matches)
                    frame = add_alert_banner(frame, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
                else:
                    st.info("No matches found.")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

elif mode == "Video Upload":
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())

        if st.button("Run Detection on Video"):
            if not query_embeddings:
                st.warning("No registered persons to detect.")
            else:
                cap = cv2.VideoCapture(tfile.name)
                frame_placeholder = st.empty()
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    # Process every 5th frame for speed
                    if frame_count % 5 == 0:
                        matches = face_engine.detect_and_match(frame, query_embeddings)
                        if matches:
                            frame = face_engine.draw_matches(frame, matches)
                            frame = add_alert_banner(frame, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
                            for match in matches:
                                db_manager.log_detection(
                                    match['person_name'], "video_upload", "Demo Video",
                                    match['similarity'], frame
                                )
                        
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                cap.release()

st.markdown("---")
st.subheader("📋 Detection Logs")
if st.button("🔄 Refresh Logs"):
    pass
    
log_file = db_manager.detection_log_file
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    if not df.empty:
        df["Confidence"] = df["similarity"].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            df.tail(10)[['timestamp', 'person_name', 'camera_name', 'Confidence']].iloc[::-1],
            use_container_width=True, 
            hide_index=True
        )
        
        if st.button("💾 Export Log"):
            out = db_manager.export_detection_report()
            if out:
                st.success(f"Exported to {out}")
    else:
        st.info("No detections log yet.")
else:
    st.info("No detections log yet.")