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


DEMO_INPUT_DIR = "inputs"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


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


@st.cache_data(show_spinner=False)
def load_demo_media():
    """Load demo media paths from the inputs directory."""
    images = []
    videos = []

    if not os.path.exists(DEMO_INPUT_DIR):
        return {"images": images, "videos": videos}

    for root, _, files in os.walk(DEMO_INPUT_DIR):
        for name in files:
            path = os.path.join(root, name)
            lower_name = name.lower()
            if lower_name.endswith(IMAGE_EXTENSIONS):
                images.append(path)
            elif lower_name.endswith(VIDEO_EXTENSIONS):
                videos.append(path)

    images.sort()
    videos.sort()
    return {"images": images, "videos": videos}

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

    st.subheader("🎞️ Demo Library")
    if st.button("Refresh Demo Files"):
        load_demo_media.clear()
        st.rerun()
    demo_media = load_demo_media()
    st.caption(f"Images: {len(demo_media['images'])}")
    st.caption(f"Videos: {len(demo_media['videos'])}")
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
demo_media = load_demo_media()

if mode == "Image Upload":
    st.markdown("<h3 style='text-align:center;'>Image Detection Workspace</h3>", unsafe_allow_html=True)
    image_source = st.radio("Image Source", ["Upload Image", "Demo Images"], horizontal=True)
    frame = None
    image_label = ""

    if image_source == "Upload Image":
        uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            file_bytes = uploaded_image.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image_label = "Uploaded Image"
    else:
        st.write("Choose from demo images")
        if demo_media["images"]:
            gallery = st.container(height=380, border=True)
            with gallery:
                for path in demo_media["images"]:
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.image(path, width=130)
                    with cols[1]:
                        st.write(os.path.basename(path))
                        if st.button("Use This Image", key=f"use_img_{path}"):
                            st.session_state["selected_demo_image"] = path

            selected_demo_image = st.session_state.get("selected_demo_image")
            if selected_demo_image and os.path.exists(selected_demo_image):
                frame = cv2.imread(selected_demo_image)
                image_label = f"Demo Image: {os.path.basename(selected_demo_image)}"
        else:
            st.info("No demo images found in inputs/. Add files to inputs/ and click Refresh Demo Files.")

    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=image_label, use_container_width=True)

        if st.button("Run Detection"):
            if not query_embeddings:
                st.warning("No registered persons to detect.")
            else:
                matches = face_engine.detect_and_match(frame, query_embeddings)
                if matches:
                    st.success(f"✅ Found {len(matches)} match(es)!")
                    frame_annotated = face_engine.draw_matches(frame, matches)
                    frame_annotated = add_alert_banner(frame_annotated, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
                    
                    # Log each detection
                    for match in matches:
                        db_manager.log_detection(
                            match['person_name'], 
                            "image_upload",
                            image_label,
                            match['similarity'], 
                            frame_annotated,
                            bbox=match['bbox']
                        )
                    
                    st.image(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB), caption="Result - Matches Found", use_container_width=True)
                    
                    # Show match details
                    st.subheader("📍 Detection Details")
                    for i, match in enumerate(matches):
                        with st.expander(f"{i+1}. {match['person_name']} - {match['similarity']:.1%} confidence"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                bbox = match['bbox']
                                st.write(f"**Location:** X: {bbox[0]}-{bbox[2]}, Y: {bbox[1]}-{bbox[3]}")
                                st.write(f"**Confidence:** {match['similarity']:.1%}")
                                st.write(f"**Size:** {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} pixels")
                else:
                    st.info("No matches found in this image.")

elif mode == "Video Upload":
    st.markdown("<h3 style='text-align:center;'>Video Detection Workspace</h3>", unsafe_allow_html=True)
    video_source = st.radio("Video Source", ["Upload Video", "Demo Videos"], horizontal=True)
    selected_video_path = None
    source_label = ""

    if video_source == "Upload Video":
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            selected_video_path = tfile.name
            source_label = "Uploaded Video"
    else:
        st.write("Choose from demo videos")
        if demo_media["videos"]:
            playlist = st.container(height=320, border=True)
            with playlist:
                for path in demo_media["videos"]:
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.write(os.path.basename(path))
                    with cols[1]:
                        if st.button("Select", key=f"use_vid_{path}"):
                            st.session_state["selected_demo_video"] = path

            selected_demo_video = st.session_state.get("selected_demo_video")
            if selected_demo_video and os.path.exists(selected_demo_video):
                selected_video_path = selected_demo_video
                source_label = f"Demo Video: {os.path.basename(selected_demo_video)}"
                st.video(selected_demo_video)
        else:
            st.info("No demo videos found in inputs/. Add files to inputs/ and click Refresh Demo Files.")

    if selected_video_path:
        st.info(source_label)

        if st.button("Run Detection on Video"):
            if not query_embeddings:
                st.warning("No registered persons to detect.")
            else:
                cap = cv2.VideoCapture(selected_video_path)
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    # Process every 5th frame for speed
                    if frame_count % 5 == 0:
                        matches = face_engine.detect_and_match(frame, query_embeddings)
                        if matches:
                            frame_annotated = face_engine.draw_matches(frame, matches)
                            frame_annotated = add_alert_banner(frame_annotated, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
                            
                            # Log each detection with bbox info
                            for match in matches:
                                db_manager.log_detection(
                                    match['person_name'], 
                                    f"video_frame_{frame_count}", 
                                    source_label,
                                    match['similarity'], 
                                    frame_annotated,
                                    bbox=match['bbox']
                                )
                        else:
                            frame_annotated = frame
                        
                        frame_placeholder.image(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                        status_text.text(f"Processing: Frame {frame_count}/{total_frames}")
                
                cap.release()
                status_text.success(f"✅ Video processing complete! {frame_count} frames processed.")

st.markdown("---")
st.subheader("📋 Detection Logs & Gallery")

log_file = db_manager.detection_log_file
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    if not df.empty:
        # Add confidence column
        df["Confidence"] = df["similarity"].apply(lambda x: f"{x:.1%}")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Summary Table", "🖼️ Detection Gallery"])
        
        with tab1:
            # Show summary table
            st.dataframe(
                df.tail(20)[['timestamp', 'person_name', 'camera_name', 'Confidence']].iloc[::-1],
                use_container_width=True, 
                hide_index=True
            )
        
        with tab2:
            # Show detailed gallery with images
            st.write(f"**Total Detections:** {len(df)}")
            
            # Display detections from newest to oldest
            for idx, row in df.iloc[::-1].iterrows():
                if os.path.exists(row['frame_path']):
                    with st.expander(f"📍 {row['person_name']} - {row['timestamp']} ({row['Confidence']})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Display detection image
                            detection_img = cv2.imread(row['frame_path'])
                            if detection_img is not None:
                                st.image(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        with col2:
                            # Display details
                            st.metric("Confidence", row['Confidence'])
                            st.write(f"**Person:** {row['person_name']}")
                            st.write(f"**Time:** {row['timestamp']}")
                            st.write(f"**Source:** {row['camera_name']}")
                            
                            # Show bbox coordinates if available
                            if pd.notna(row['bbox_x1']):
                                st.write("**Location (pixels):**")
                                st.code(f"X: {int(row['bbox_x1'])}-{int(row['bbox_x2'])}\nY: {int(row['bbox_y1'])}-{int(row['bbox_y2'])}\nSize: {int(row['bbox_x2'])-int(row['bbox_x1'])}x{int(row['bbox_y2'])-int(row['bbox_y1'])} px", language=None)
        
        # Export button
        if st.button("💾 Export Detection Report"):
            out = db_manager.export_detection_report()
            if out:
                st.success(f"Exported to {out}")
    else:
        st.info("No detections log yet.")
else:
    st.info("No detections log yet.")