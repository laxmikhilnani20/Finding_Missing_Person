"""
CCTV-Based Missing Person Detection System
Real-time face recognition across multiple IP camera streams
"""

import streamlit as st
import cv2
import time
import threading
from datetime import datetime
from PIL import Image
import pandas as pd

# Import custom modules
from src.face_recognition_engine import FaceRecognitionEngine
from src.ip_camera_manager import IPCameraManager
from src.database_manager import DatabaseManager
from src.utils import resize_frame, add_timestamp_overlay, add_alert_banner, validate_ip_url

# Page configuration
st.set_page_config(
    page_title="CCTV Missing Person Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert > div {
        padding: 1rem;
    }
    .detection-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
    .camera-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    .status-active {
        background-color: #00cc00;
        color: white;
    }
    .status-inactive {
        background-color: #999999;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.face_engine = None
    st.session_state.camera_manager = None
    st.session_state.db_manager = DatabaseManager()
    st.session_state.monitoring = False
    st.session_state.query_embeddings = {}
    st.session_state.detection_count = 0
    st.session_state.last_detection = None


def initialize_system():
    """Initialize face recognition engine and camera manager"""
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing Face Recognition Engine..."):
            st.session_state.face_engine = FaceRecognitionEngine(similarity_threshold=0.65)
            st.session_state.camera_manager = IPCameraManager()
            st.session_state.initialized = True
            
            # Load saved camera configuration
            saved_cameras = st.session_state.db_manager.load_camera_config()
            for cam in saved_cameras:
                st.session_state.camera_manager.add_camera(
                    cam['id'], cam['name'], cam['url']
                )


def load_missing_persons():
    """Load missing persons and generate embeddings"""
    persons = st.session_state.db_manager.get_missing_persons()
    st.session_state.query_embeddings = {}
    
    for name, image_path in persons.items():
        try:
            image = Image.open(image_path)
            embedding = st.session_state.face_engine.encode_face(image)
            if embedding is not None:
                st.session_state.query_embeddings[name] = embedding
        except Exception as e:
            st.warning(f"⚠️ Could not load embedding for {name}: {e}")


def sidebar_camera_management():
    """Sidebar section for camera management"""
    st.sidebar.header("📹 Camera Management")
    
    # Add new camera
    with st.sidebar.expander("➕ Add New Camera", expanded=False):
        cam_name = st.text_input("Camera Name", placeholder="e.g., Entrance")
        cam_url = st.text_input("IP Camera URL", placeholder="http://192.168.1.100:8080/video")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Connection"):
                if cam_url and st.session_state.camera_manager:
                    with st.spinner("Testing..."):
                        if st.session_state.camera_manager.test_connection(cam_url):
                            st.success("✅ Connected!")
                        else:
                            st.error("❌ Failed to connect")
        
        with col2:
            if st.button("Add Camera"):
                if cam_name and cam_url:
                    if validate_ip_url(cam_url):
                        cam_id = f"cam_{int(time.time())}"
                        if st.session_state.camera_manager.add_camera(cam_id, cam_name, cam_url):
                            st.success(f"✅ Added {cam_name}")
                            
                            # Save configuration
                            cameras = st.session_state.camera_manager.get_all_camera_info()
                            camera_list = [
                                {'id': cid, 'name': info['name'], 'url': info['url']}
                                for cid, info in cameras.items()
                            ]
                            st.session_state.db_manager.save_camera_config(camera_list)
                            st.rerun()
                        else:
                            st.error("❌ Failed to add camera")
                    else:
                        st.error("❌ Invalid URL format")
                else:
                    st.warning("⚠️ Please enter both name and URL")
    
    # Display existing cameras
    st.sidebar.subheader("Active Cameras")
    if st.session_state.camera_manager:
        cameras = st.session_state.camera_manager.get_all_camera_info()
        
        if cameras:
            for cam_id, info in cameras.items():
                status = "🟢 Active" if info['is_active'] else "🔴 Inactive"
                st.sidebar.text(f"{info['name']}: {status}")
                st.sidebar.text(f"   FPS: {info['fps']}")
                
                if st.sidebar.button(f"Remove {info['name']}", key=f"remove_{cam_id}"):
                    st.session_state.camera_manager.remove_camera(cam_id)
                    
                    # Update saved configuration
                    cameras = st.session_state.camera_manager.get_all_camera_info()
                    camera_list = [
                        {'id': cid, 'name': i['name'], 'url': i['url']}
                        for cid, i in cameras.items()
                    ]
                    st.session_state.db_manager.save_camera_config(camera_list)
                    st.rerun()
                
                st.sidebar.divider()
        else:
            st.sidebar.info("No cameras added yet")


def sidebar_missing_persons():
    """Sidebar section for missing persons management"""
    st.sidebar.header("👤 Missing Persons")
    
    # Upload new missing person
    with st.sidebar.expander("➕ Add Missing Person", expanded=False):
        person_name = st.text_input("Person Name", placeholder="e.g., John Doe")
        person_image = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
        
        if st.button("Add Person"):
            if person_name and person_image:
                image_path = st.session_state.db_manager.add_missing_person(
                    person_name, person_image
                )
                if image_path:
                    st.success(f"✅ Added {person_name}")
                    load_missing_persons()
                    st.rerun()
                else:
                    st.error("❌ Failed to add person")
            else:
                st.warning("⚠️ Please provide name and photo")
    
    # Display missing persons
    st.sidebar.subheader("Registered Persons")
    persons = st.session_state.db_manager.get_missing_persons()
    
    if persons:
        for name, image_path in persons.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.text(f"👤 {name}")
            with col2:
                if st.button("🗑️", key=f"del_{name}"):
                    st.session_state.db_manager.remove_missing_person(name)
                    load_missing_persons()
                    st.rerun()
            
            # Show thumbnail
            try:
                img = Image.open(image_path)
                st.sidebar.image(img, width=150)
            except:
                pass
            
            st.sidebar.divider()
    else:
        st.sidebar.info("No persons registered yet")


def main_monitoring_interface():
    """Main monitoring interface"""
    st.title("🔍 CCTV Missing Person Detection System")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Monitoring", disabled=st.session_state.monitoring):
            if not st.session_state.query_embeddings:
                st.error("❌ Please add missing persons first!")
            elif not st.session_state.camera_manager.cameras:
                st.error("❌ Please add cameras first!")
            else:
                st.session_state.camera_manager.start_all()
                st.session_state.monitoring = True
                st.success("✅ Monitoring started")
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.monitoring):
            st.session_state.camera_manager.stop_all()
            st.session_state.monitoring = False
            st.info("⏹️ Monitoring stopped")
            st.rerun()
    
    with col3:
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.65,
            step=0.05,
            key="threshold"
        )
        if st.session_state.face_engine:
            st.session_state.face_engine.set_similarity_threshold(threshold)
    
    with col4:
        if st.button("📊 Export Report"):
            report_path = st.session_state.db_manager.export_detection_report()
            if report_path:
                st.success(f"✅ Report exported!")
    
    # Statistics
    st.metric(
        label="Total Detections",
        value=st.session_state.detection_count,
        delta="New" if st.session_state.last_detection else None
    )
    
    # Live feeds
    if st.session_state.monitoring:
        st.subheader("📡 Live Camera Feeds")
        
        cameras = st.session_state.camera_manager.get_all_camera_info()
        
        if not cameras:
            st.warning("⚠️ No active cameras")
            return
        
        # Create columns for camera feeds
        num_cameras = len(cameras)
        cols_per_row = 2
        
        camera_ids = list(cameras.keys())
        
        for i in range(0, num_cameras, cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < num_cameras:
                    cam_id = camera_ids[i + j]
                    cam_info = cameras[cam_id]
                    
                    with col:
                        st.write(f"**{cam_info['name']}**")
                        frame_placeholder = st.empty()
                        status_placeholder = st.empty()
                        
                        # Get frame from camera
                        frame = st.session_state.camera_manager.get_frame(cam_id)
                        
                        if frame is not None:
                            # Detect faces
                            matches = st.session_state.face_engine.detect_and_match(
                                frame,
                                st.session_state.query_embeddings
                            )
                            
                            # Draw matches
                            if matches:
                                frame = st.session_state.face_engine.draw_matches(frame, matches)
                                frame = add_alert_banner(
                                    frame,
                                    f"🚨 {matches[0]['person_name'].upper()} DETECTED!"
                                )
                                
                                # Log detection
                                for match in matches:
                                    st.session_state.db_manager.log_detection(
                                        match['person_name'],
                                        cam_id,
                                        cam_info['name'],
                                        match['similarity'],
                                        frame
                                    )
                                    st.session_state.detection_count += 1
                                    st.session_state.last_detection = datetime.now()
                                
                                status_placeholder.error(
                                    f"🚨 ALERT: {matches[0]['person_name']} detected! "
                                    f"Confidence: {matches[0]['similarity']:.1%}"
                                )
                            else:
                                status_placeholder.success("🔍 Searching...")
                            
                            # Add timestamp overlay
                            frame = add_timestamp_overlay(frame, cam_info['name'])
                            
                            # Display frame
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_resized = resize_frame(frame_rgb, max_width=640)
                            frame_placeholder.image(frame_resized, use_column_width=True)
                        else:
                            status_placeholder.warning("⚠️ No frame available")
        
        # Auto-refresh
        time.sleep(0.5)
        st.rerun()
    else:
        st.info("ℹ️ Click 'Start Monitoring' to begin detection")


def detection_log_tab():
    """Display detection log"""
    st.subheader("📝 Detection Log")
    
    log_df = st.session_state.db_manager.get_detection_log(limit=50)
    
    if not log_df.empty:
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", len(log_df))
        with col2:
            unique_persons = log_df['person_name'].nunique()
            st.metric("Unique Persons", unique_persons)
        with col3:
            avg_confidence = log_df['similarity'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Display log table
        st.dataframe(
            log_df[['timestamp', 'person_name', 'camera_name', 'similarity']]
        )
        
        # Show recent detection images
        st.subheader("Recent Detections")
        recent = log_df.tail(6)
        
        cols = st.columns(3)
        for idx, (_, row) in enumerate(recent.iterrows()):
            col = cols[idx % 3]
            with col:
                try:
                    img = Image.open(row['frame_path'])
                    st.image(img, caption=f"{row['person_name']} - {row['timestamp']}")
                except:
                    st.warning(f"Could not load image: {row['frame_path']}")
    else:
        st.info("No detections logged yet")


def main():
    """Main application"""
    # Initialize system
    initialize_system()
    
    # Load missing persons
    if st.session_state.query_embeddings == {}:
        load_missing_persons()
    
    # Sidebar
    sidebar_camera_management()
    st.sidebar.divider()
    sidebar_missing_persons()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Live Monitoring", "📝 Detection Log", "ℹ️ About"])
    
    with tab1:
        main_monitoring_interface()
    
    with tab2:
        detection_log_tab()
    
    with tab3:
        st.header("About This System")
        st.markdown("""
        ### 🔍 CCTV-Based Missing Person Detection System
        
        This system uses **real-time face recognition** to detect missing persons across multiple IP camera streams.
        
        **Features:**
        - 📹 Multi-camera support via IP addresses
        - 👤 Multiple missing person profiles
        - 🎯 Real-time face detection and matching
        - 🚨 Instant alerts with camera location
        - 📝 Comprehensive detection logging
        - 📊 Exportable reports
        
        **Tech Stack:**
        - **Face Recognition:** MTCNN + InceptionResnetV1 (FaceNet)
        - **Framework:** Streamlit
        - **Computer Vision:** OpenCV
        - **Deployment:** Docker
        
        **How to Use:**
        1. Add IP cameras with their stream URLs
        2. Register missing persons with their photos
        3. Click "Start Monitoring" to begin detection
        4. System will alert when a match is found
        
        **Developed for PBL Project - 2025**
        """)


if __name__ == "__main__":
    main()
