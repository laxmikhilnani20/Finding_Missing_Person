# Finding Missing Person - Agent Documentation

This document explains the architecture, structure, and functionality of the "Finding Missing Person" application. The project is designed to be an end-to-end system that identifies missing persons from live camera feeds.

## 1. Architecture Overview
The project is built as a monolithic Streamlit application that serves as both the frontend dashboard and backend orchestrator. It uses a multithreaded architecture to handle concurrent video streams from multiple IP cameras while running deep learning models for facial recognition in real-time. Data is managed via local file storage (JSON, CSV, and image files) rather than a relational database.

## 2. Frontend (Streamlit Dashboard - `app.py`)
The frontend is driven by Streamlit and is split into two main sections:
* **Sidebar (Control Panel):** Allows users to dynamically add/remove IP camera streams by providing a name and URL (HTTP/RTSP). It includes a functional "Test Connection" button. Here, users can also upload photos to register new missing persons and adjust the model's similarity confidence threshold.
* **Main Dashboard:** The core operational view. It contains a "Start/Stop Monitoring" toggle. When active, it runs an infinite loop reading the latest frames from the active cameras, passing them to the face recognition engine, and rendering the annotated live feeds onto the UI using `st.empty()` placeholders. The dashboard also displays a live tabular log of recent detections.

## 3. IP Webcam Integration (`src/ip_camera_manager.py`)
To handle high-latency or multiple video streams without freezing the main application, the camera management is decoupled:
* **Threaded Capturing:** The `IPCamera` class leverages `cv2.VideoCapture` to connect to streams. Critically, it launches a background daemon thread for each camera to continuously read frames.
* **Queue System:** The thread pushes frames into a `queue.Queue` with a maximum size of 2. This frame-dropping mechanism prevents buffer bloat and ensures the main Streamlit loop always retrieves the most recent frame, avoiding stream latency/lag.
* **Manager Class:** `IPCameraManager` provides an interface to easily start, stop, and fetch frames (`get_frame()`) for all registered cameras.

## 4. Face Recognition Engine (`src/face_recognition_engine.py`)
The brain of the operation, powered by PyTorch and `facenet-pytorch`:
* **Detection (MTCNN):** Uses the Multi-task Cascaded Convolutional Networks (MTCNN) to scan a frame and extract the bounding boxes (coordinates) of any human faces present.
* **Encoding (InceptionResnetV1):** Once a face is cropped, it is passed through an InceptionResnetV1 model (pretrained on the `vggface2` dataset) to generate a 512-dimensional numerical embedding/vector.
* **Matching:** In `app.py`, all registered missing persons' photos are pre-processed into embeddings (`query_embeddings`). When analyzing a live video frame, the engine extracts the live face embeddings and calculates the **Cosine Similarity** (`F.cosine_similarity`) against the database embeddings. If the score exceeds the configurable threshold (default `0.65`), it flags a match and annotates the frame with red bounding boxes and the person's name.

## 5. Backend Data Storage (`src/database_manager.py`)
Instead of SQL, the app relies on a lightweight, file-based storage architecture managed by `DatabaseManager`:
* **Profiles:** Registered missing persons' uploaded photos are saved into a specific folder structure (`data/missing_persons/<Name>/profile.jpg`).
* **Configurations:** Active camera setups are stored in `config/cameras.json` to persist across app reboots.
* **Detection Logs:** When a missing person is identified in a frame, the manager saves the raw frame screenshot into a `data/detections/` folder and appends the metadata (timestamp, person name, camera name, similarity score) to a `detection_log.csv` file, which is actively read by Streamlit to display the log table.

## 6. Utilities (`src/utils.py`)
This script supplies non-blocking OpenCV (`cv2`) overlay functions that trigger right before displaying the stream frames to the user:
* `add_timestamp_overlay`: Adds the camera name and current live timestamp to the top of the video feed.
* `add_alert_banner`: If a person is matched, it injects a highly visible red "🚨 PERSON DETECTED!" banner at the bottom of the frame before saving/displaying it.

## Execution Flow
1. **Startup:** Streamlit runs `app.py`. It initializes the components, loads previously saved cameras from `cameras.json`, and pre-computes PyTorch embeddings for all images sitting in `data/missing_persons/`.
2. **Setup:** The user adds RTSP/HTTP camera URLs via the sidebar. `IPCameraManager` creates background threads that constantly pull frames from the streams into short queues.
3. **Monitoring:** When the user hits "Start Monitoring", Streamlit enters a fast `while` loop.
4. **Processing Pipeline:** For every iteration of the loop, for every active camera:
   * It pulls the latest frame from the camera's queue.
   * Sends the frame to `FaceRecognitionEngine.detect_and_match()`.
   * The Engine finds faces and calculates their similarity to the missing persons.
   * If a match is found, `DatabaseManager` logs the event to a CSV and drops the screenshot into a folder. `utils.py` draws an alert banner on the image.
   * Finally, Streamlit pushes the resulting annotated image to the dashboard UI.