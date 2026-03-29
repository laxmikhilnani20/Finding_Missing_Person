import streamlit as st
import cv2
import time
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import torch

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
    cam_manager = IPCameraManager()
    db_manager = DatabaseManager()
    
    # Load previously saved cameras from json
    saved_cameras = db_manager.load_camera_config()
    for cam in saved_cameras:
        cam_manager.add_camera(cam['id'], cam['name'], cam['url'])
        
    return engine, cam_manager, db_manager

face_engine, camera_manager, db_manager = init_system()

# Monitor State Initialization
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

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

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    # --- 1. Camera Management ---
    st.subheader("📹 Add External IP Camera")
    cam_name = st.text_input("Name (e.g., Entrance, Phone)")
    cam_url = st.text_input("Stream URL (e.g. http://192.168.../video)")
    
    col_t, col_a = st.columns(2)
    with col_t:
        test_cam = st.button("Test Connection")
    with col_a:
        add_cam = st.button("Add Camera")

    if test_cam:
        if cam_url and validate_ip_url(cam_url):
            st.info(f"Connecting to {cam_url} ...")
            test_cap = cv2.VideoCapture(cam_url)
            if test_cap.isOpened():
                st.success("✅ Success! Camera is reachable.")
                test_cap.release()
            else:
                st.error("❌ Failed. The server cannot reach this IP address.")
        else:
            st.warning("Valid URL required.")
            
    if add_cam:
        if cam_name and cam_url and validate_ip_url(cam_url):
            cam_id = f"cam_{int(time.time())}"
            if camera_manager.add_camera(cam_id, cam_name, cam_url):
                # Save config
                cams = [{"id": cid, "name": info["name"], "url": info["url"]} 
                        for cid, info in camera_manager.get_all_camera_info().items()]
                db_manager.save_camera_config(cams)
                st.success(f"Added {cam_name}!")
                st.rerun()
            else:
                st.error("Failed to connect.")
        else:
            st.error("Invalid Name or URL.")

    # Show active cameras & removal option
    active_cams = camera_manager.get_all_camera_info()
    if active_cams:
        st.write("**Active Cameras**")
        for cid, info in active_cams.items():
            cc1, cc2 = st.columns([4, 1])
            cc1.caption(f"{info['name']} ({'🟢' if info['is_active'] else '🔴'})")
            if cc2.button("🗑️", key=f"del_{cid}", help="Remove Camera"):
                camera_manager.remove_camera(cid)
                cams_updated = [{"id": _id, "name": _info["name"], "url": _info["url"]} 
                                for _id, _info in camera_manager.get_all_camera_info().items()]
                db_manager.save_camera_config(cams_updated)
                st.rerun()

    st.markdown("---")
    
    # --- 2. Person Management ---
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
    
    # --- 3. System Settings ---
    st.subheader("📊 Model Settings")
    threshold = st.slider("Match Confidence Threshold", 0.0, 1.0, 0.65, 0.05)
    if face_engine.similarity_threshold != threshold:
        face_engine.set_similarity_threshold(threshold)

# --- Main Dashboard ---
st.title("🔍 CCTV Missing Person Detection System")

# Refresh embeddings
query_embeddings = load_embeddings()

col_main, col_logs = st.columns([3, 1])

with col_main:
    st.subheader("📹 Live IP Camera Feeds")
    
    # Start/Stop Monitor Controls
    ctrl1, ctrl2 = st.columns([1, 4])
    if not st.session_state.monitoring:
        if ctrl1.button("▶️ Start Monitoring", use_container_width=True):
            if camera_manager.start_all():
                st.session_state.monitoring = True
                st.rerun()
            else:
                st.error("Failed to start. Ensure you have added active cameras.")
    else:
        if ctrl1.button("⏹️ Stop Monitoring", use_container_width=True, type="primary"):
            st.session_state.monitoring = False
            camera_manager.stop_all()
            st.rerun()
            
    # Video Feeds Section
    if st.session_state.monitoring:
        cams_info = camera_manager.get_all_camera_info()
        if not cams_info:
            st.warning("No linked cameras!")
            st.session_state.monitoring = False
            st.rerun()
            
        st.caption("Monitoring Live Streams...")
        
        # We create a placeholder grid for the cameras
        cam_placeholders = {}
        cols = st.columns(len(cams_info) if len(cams_info) <= 2 else 2) # max 2 per row
        col_idx = 0
        
        for cid, info in cams_info.items():
            with cols[col_idx % 2]:
                st.write(f"**{info['name']}**")
                cam_placeholders[cid] = st.empty()
            col_idx += 1
            
        # The Infinite Stream Loop
        while st.session_state.monitoring:
            for cid, info in cams_info.items():
                frame = camera_manager.get_frame(cid)
                
                if frame is not None:
                    # 1. Detection Process
                    matches = face_engine.detect_and_match(frame, query_embeddings)
                    
                    if matches:
                        frame = face_engine.draw_matches(frame, matches)
                        frame = add_alert_banner(frame, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
                        
                        # Database logging logic
                        for match in matches:
                            db_manager.log_detection(
                                match['person_name'], cid, info['name'],
                                match['similarity'], frame
                            )
                            
                    # 2. Frame Processing for Display
                    frame = add_timestamp_overlay(frame, info['name'])
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    cam_placeholders[cid].image(frame_rgb, channels="RGB", use_container_width=True)
                    
            time.sleep(0.04)

with col_logs:
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