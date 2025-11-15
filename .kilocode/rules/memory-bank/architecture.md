# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Flask Web Server (Port 5000)              │
│                    + Socket.IO WebSocket                 │
└────────────┬───────────────────────────┬─────────────────┘
             │                           │
    ┌────────▼────────┐         ┌───────▼──────────┐
    │   REST API      │         │   WebSocket      │
    │   Endpoints     │         │   (Real-time)    │
    └────────┬────────┘         └────────┬─────────┘
             └──────────┬────────────────┘
                        │
             ┌──────────▼──────────────────┐
             │  Application State Manager  │
             │  (Global AppState Object)   │
             └──────────┬──────────────────┘
                        │
        ┌───────────────┼───────────────────┐
        │               │                   │
┌───────▼─────┐  ┌─────▼──────┐  ┌────────▼────────┐
│  IP Camera  │  │    Face    │  │    Database     │
│   Manager   │  │ Recognition│  │     Manager     │
└─────────────┘  │   Engine   │  └─────────────────┘
                 └────────────┘
```

## Source Code Paths

### Core Application
- **`/app_flask.py`** (392 lines)
  - Main Flask application entry point
  - Route definitions and WebSocket handlers
  - Application state management
  - Thread orchestration for camera processing

### Core Modules (`/src/`)
- **`/src/face_recognition_engine.py`**
  - FaceRecognitionEngine class
  - MTCNN face detection
  - InceptionResnetV1 face encoding
  - Cosine similarity matching
  - Bounding box drawing
  
- **`/src/ip_camera_manager.py`** (239 lines)
  - IPCamera class (individual camera handler)
  - IPCameraManager class (multi-camera orchestrator)
  - Thread-based frame capture
  - Queue-based frame buffering
  
- **`/src/database_manager.py`** (240 lines)
  - DatabaseManager class
  - File-based storage operations
  - Missing person management
  - Detection logging (CSV)
  - Camera configuration (JSON)
  
- **`/src/utils.py`** (206 lines)
  - Helper functions
  - Frame processing utilities
  - Overlay rendering
  - URL validation

### Frontend (`/templates/` & `/static/`)
- **`/templates/index.html`** (321 lines)
  - Single-page dashboard
  - Modal dialogs for camera/person management
  - Tab-based interface (Monitoring, Logs, About)
  
- **`/static/js/app.js`** (521 lines)
  - Socket.IO client
  - API calls to Flask backend
  - Real-time frame updates
  - Alert management
  
- **`/static/css/style.css`**
  - Dark theme styling
  - Responsive grid layouts
  - Animation effects

### Configuration & Deployment
- **`/Dockerfile_flask`**
  - Python 3.10 slim base
  - System dependencies (libgl1, libglib2.0-0, etc.)
  - Python package installation
  
- **`/docker_compose_flask.yml`**
  - Service definition
  - Port mapping (5000:5000)
  - Volume mounts (./data, ./config)
  
- **`/requirements_flask.txt`**
  - All Python dependencies with pinned versions

### Data Storage (`/data/` - gitignored)
- **`/data/missing_persons/{name}/profile.jpg`**
  - Person profile images
  
- **`/data/detections/{person}_{camera}_{timestamp}.jpg`**
  - Detection snapshots
  
- **`/data/detection_log.csv`**
  - Columns: timestamp, person_name, camera_id, camera_name, similarity, frame_path
  
- **`/data/models/`**
  - Cached PyTorch models (MTCNN, InceptionResnetV1)

### Configuration (`/config/`)
- **`/config/cameras.json`**
  - Array of camera objects: {id, name, url}

## Key Technical Decisions

### 1. Multi-Threading Architecture
**Decision**: Each camera runs in separate daemon thread with queue-based frame management.

**Rationale**:
- Non-blocking frame capture
- Independent camera processing
- Automatic cleanup on thread exit
- Prevents one slow camera from blocking others

**Implementation**:
```python
# In IPCamera class
self.frame_queue = queue.Queue(maxsize=2)
self.thread = threading.Thread(target=self._capture_frames, daemon=True)
```

### 2. Frame Skip Counter
**Decision**: Process every 2nd frame for AI detection (but display all frames).

**Rationale**:
- Camera capture faster than AI processing
- Prevents frame queue backlog
- Maintains smooth video display
- Reduces CPU usage

**Implementation**:
```python
frame_skip_counter += 1
should_detect = (frame_skip_counter % 2 == 0) or len(app_state.query_embeddings) == 0
```

### 3. WebSocket for Real-Time Updates
**Decision**: Use Socket.IO instead of HTTP polling.

**Rationale**:
- Zero-latency notifications
- Bidirectional communication
- Automatic reconnection
- Efficient bandwidth usage

**Events**:
- `frame_update`: Camera frame data
- `detection_alert`: Person detected notification
- `connect`/`disconnect`: Connection lifecycle

### 4. File-Based Storage
**Decision**: Use filesystem + CSV/JSON instead of database.

**Rationale**:
- Simpler deployment (no DB setup)
- Sufficient for current scale
- Easy data portability
- Docker volume mounts work seamlessly

**Trade-off**: Not suitable for large-scale enterprise deployment.

### 5. Global Application State
**Decision**: Use single AppState class instance.

**Rationale**:
- Centralized state management
- Easy access across routes
- Thread-safe with proper locking (if needed)
- Simplifies initialization

**Structure**:
```python
class AppState:
    initialized, face_engine, camera_manager, db_manager,
    monitoring, query_embeddings, detection_count, last_detection, monitoring_threads
```

### 6. Docker Containerization
**Decision**: Bundle all dependencies in Docker image.

**Rationale**:
- Zero local setup required
- Consistent environment across platforms
- Easy deployment and scaling
- Isolation from host system

## Design Patterns

### 1. Manager Pattern
Used for IPCameraManager and DatabaseManager:
- Centralized control of multiple resources
- Abstraction of complex operations
- State management

### 2. Singleton-like State
AppState acts as singleton:
- One instance shared across application
- Global access to system components

### 3. Producer-Consumer Pattern
Camera threads (producer) → Frame queue → Processing threads (consumer)
- Decouples capture from processing
- Handles speed mismatches

### 4. Observer Pattern (via WebSocket)
Backend events → Socket.IO → Multiple frontend clients
- Real-time updates to all connected clients

## Component Relationships

### Initialization Flow
```
main() → initialize_system() → 
  ├─ FaceRecognitionEngine (load MTCNN + ResNet)
  ├─ IPCameraManager (empty, cameras added via API)
  ├─ DatabaseManager (create directories)
  └─ load_missing_persons() (generate embeddings)
```

### Detection Flow
```
Camera Thread (capture) →
Frame Queue →
process_camera_stream() →
  ├─ detect_and_match() →
  │   ├─ MTCNN (detect faces)
  │   ├─ ResNet (encode faces)
  │   └─ Cosine similarity (match)
  ├─ draw_matches() (annotate frame)
  ├─ log_detection() (save to DB)
  └─ emit('frame_update') → WebSocket → Frontend
```

### API Request Flow
```
Frontend API Call →
Flask Route →
  ├─ Validate input
  ├─ Call Manager method
  ├─ Update AppState
  └─ Return JSON response
```

## Critical Implementation Paths

### Adding a Camera
1. POST `/api/cameras/add` with {name, url}
2. `validate_ip_url(url)` checks format
3. `camera_manager.add_camera()` creates IPCamera instance
4. `camera.connect()` opens cv2.VideoCapture
5. `db_manager.save_camera_config()` persists to JSON
6. Frontend updates camera list via API response

### Starting Monitoring
1. POST `/api/monitoring/start`
2. Check query_embeddings and cameras exist
3. `camera_manager.start_all()` starts capture threads
4. Create processing thread for each camera
5. Each thread runs `process_camera_stream(camera_id)` loop
6. Threads emit frames via Socket.IO

### Face Detection & Matching
1. Get frame from camera queue
2. Convert BGR → RGB, create PIL Image
3. `mtcnn.detect()` finds face bounding boxes
4. `mtcnn.forward()` extracts face tensors
5. `resnet()` generates 512-d embeddings
6. For each face, compute cosine similarity vs all persons
7. If similarity ≥ threshold, create match record
8. Draw bounding box + label on frame
9. Log detection to CSV + save snapshot
10. Emit alert via Socket.IO

### Persisting Detection
1. `db_manager.log_detection()` called on match
2. Generate timestamp-based filename
3. `cv2.imwrite()` saves frame snapshot
4. Read existing CSV with pandas
5. Append new row with detection data
6. Write updated CSV back to disk

## Data Flow Diagram

```
IP Camera Stream
      ↓
cv2.VideoCapture (IPCamera thread)
      ↓
Queue (maxsize=2)
      ↓
get_frame() (process_camera_stream)
      ↓
PIL.Image → MTCNN → Face Tensors
      ↓
InceptionResnetV1 → 512-d Embeddings
      ↓
Cosine Similarity vs Query Embeddings
      ↓
Match Found? → draw_matches() → add_timestamp_overlay()
      ↓
base64 encode → emit('frame_update')
      ↓
Frontend Socket.IO → Update <img> src
```

## Scaling Considerations

### Current Limitations
- Single-process Flask server (not distributed)
- File-based storage (no query optimization)
- CPU-only processing by default
- Memory grows with camera count

### Horizontal Scaling (Future)
- Deploy multiple Flask instances behind load balancer
- Use Redis for shared state
- Use PostgreSQL/MongoDB for detections
- Implement camera sharding across instances

### Vertical Scaling (Current)
- Increase Docker memory limit
- Use GPU-enabled container
- Optimize frame processing rate
- Reduce camera resolution
