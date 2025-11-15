# Technology Stack

## Programming Languages
- **Python 3.10**: Backend application, AI models, image processing
- **JavaScript (ES6+)**: Frontend logic, Socket.IO client, DOM manipulation
- **HTML5**: Dashboard structure, semantic markup
- **CSS3**: Styling, animations, dark theme

## Web Framework
- **Flask 3.0.0**: Lightweight web framework for REST API
  - Route handling
  - Request/response processing
  - Static file serving
  - Template rendering (Jinja2)
  
- **Flask-SocketIO 5.3.5**: WebSocket support for real-time updates
  - Bidirectional communication
  - Event-based messaging
  - Room/namespace support
  - Automatic reconnection

## Deep Learning & Computer Vision

### Face Detection
- **MTCNN** (Multi-task Cascaded Convolutional Networks)
  - Via `facenet-pytorch 2.6.0`
  - Three-stage cascade: P-Net, R-Net, O-Net
  - Outputs: bounding boxes + facial landmarks
  - Thresholds: [0.6, 0.7, 0.7]
  - `keep_all=True` to detect multiple faces

### Face Recognition
- **InceptionResnetV1** (FaceNet)
  - Via `facenet-pytorch 2.6.0`
  - Pretrained on VGGFace2 dataset
  - Output: 512-dimensional embedding vectors
  - Model automatically downloaded on first run
  - Cached in `data/models/`

### Deep Learning Framework
- **PyTorch 2.2.2**: Neural network framework
  - Auto-detects CUDA GPU if available
  - CPU fallback for compatibility
  - Model serialization/deserialization
  - Gradient computation disabled in inference

- **torchvision 0.17.2**: Vision utilities
  - Image transformations
  - Pretrained model loading

## Computer Vision
- **OpenCV 4.9.0.80** (headless build)
  - `cv2.VideoCapture`: Stream from IP cameras
  - `cv2.imread/imwrite`: Image I/O
  - `cv2.resize`: Frame resizing
  - `cv2.rectangle`: Bounding box drawing
  - `cv2.putText`: Label rendering
  - `cv2.cvtColor`: Color space conversion (BGR ↔ RGB)
  - Headless: No GUI dependencies (Qt/GTK)

## Data Processing
- **NumPy 1.26.4**: Numerical computing
  - Array operations
  - Mathematical functions
  - Image data representation

- **Pandas 2.2.2**: Data manipulation
  - CSV reading/writing
  - DataFrame operations
  - Detection log management

- **Pillow 10.2.0**: Image processing
  - PIL.Image for face detection input
  - Format conversion
  - Image saving

## Web Server
- **Werkzeug 3.0.1**: WSGI utility library
  - Request/response objects
  - Routing
  - Debugging utilities

- **eventlet 0.33.3**: Async networking
  - WebSocket transport
  - Concurrent connection handling
  - Green thread implementation

## Frontend Libraries
- **Socket.IO Client**: WebSocket communication
  - Loaded via CDN in HTML
  - Auto-reconnection
  - Event listeners

- **Font Awesome 6.4.0**: Icon library
  - UI icons (camera, user, bell, etc.)
  - Loaded via CDN

- **Animate.css 4.1.1**: CSS animations
  - Alert animations
  - Transition effects
  - Loaded via CDN

## Containerization
- **Docker**: Container platform
  - Base image: `python:3.10-slim`
  - Multi-stage build not used (simple single stage)
  
- **Docker Compose**: Multi-container orchestration
  - Service definition
  - Network configuration
  - Volume management

## Development Setup

### Local Development (without Docker)
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements_flask.txt

# Run application
python app_flask.py
```

### Docker Development
```bash
# Build image
docker-compose -f docker_compose_flask.yml build

# Run container
docker-compose -f docker_compose_flask.yml up

# Run in background
docker-compose -f docker_compose_flask.yml up -d

# Stop container
docker-compose -f docker_compose_flask.yml down
```

## Technical Constraints

### Hardware Requirements
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU
- **GPU**: Optional (CUDA-compatible NVIDIA GPU)
- **Storage**: ~500MB for models + variable for detections

### Software Requirements
- **Docker Desktop**: Latest version
- **Operating System**: Windows 10+, macOS 10.15+, Linux (any recent distro)
- **Network**: Access to camera stream URLs
- **Browser**: Chrome/Firefox/Safari/Edge (WebSocket support required)

### Performance Constraints
- Face detection: ~30-60ms per frame (CPU)
- Face encoding: ~50-100ms per face (CPU)
- Total latency: <200ms from capture to alert
- Memory per camera: ~200-300MB
- Network bandwidth: ~1-5 Mbps per HD camera stream

### Docker Constraints
- Container runs as single process
- No GPU access by default (need nvidia-docker for GPU)
- Volume mounts required for data persistence
- Port 5000 must be available
- Network bridge driver for isolation

## Dependencies

### Python Packages (requirements_flask.txt)
```
Flask==3.0.0
Flask-SocketIO==5.3.5
python-socketio==5.10.0
eventlet==0.33.3

opencv-python-headless==4.9.0.80
facenet-pytorch==2.6.0
torch==2.2.2
torchvision==0.17.2

numpy==1.26.4
pandas==2.2.2
pillow==10.2.0

python-dateutil==2.9.0
requests==2.32.3
Werkzeug==3.0.1
```

### System Dependencies (in Dockerfile)
```
libgl1           # OpenGL library
libglib2.0-0     # GLib utilities
libsm6           # X11 Session Management
libxext6         # X11 extensions
libxrender1      # X11 rendering
libgomp1         # OpenMP parallel processing
libgstreamer1.0-0               # GStreamer multimedia
libgstreamer-plugins-base1.0-0  # GStreamer plugins
wget             # Network downloader
```

## Tool Usage Patterns

### Flask Route Pattern
```python
@app.route('/api/endpoint', methods=['POST'])
def handler():
    data = request.json
    # Process
    return jsonify({'success': True, 'data': result})
```

### WebSocket Event Pattern
```python
@socketio.on('event_name')
def handle_event(data):
    emit('response_event', {'data': data})
```

### Camera Thread Pattern
```python
def process_camera_stream(camera_id):
    while app_state.monitoring:
        frame = app_state.camera_manager.get_frame(camera_id)
        # Process frame
        socketio.emit('frame_update', data)
```

### Database Pattern
```python
# CSV logging
df = pd.read_csv(self.detection_log_file)
new_row = pd.DataFrame([log_entry])
df = pd.concat([df, new_row], ignore_index=True)
df.to_csv(self.detection_log_file, index=False)

# JSON config
with open(config_file, 'w') as f:
    json.dump(data, f, indent=2)
```

### Face Recognition Pattern
```python
# Detect
boxes, _ = self.mtcnn.detect(pil_image)

# Extract faces
face_tensors = self.mtcnn.forward(pil_image, save_path=None)

# Encode
with torch.no_grad():
    embeddings = self.resnet(face_tensors)

# Match
similarity = F.cosine_similarity(embedding1, embedding2).item()
```

## Version Control
- **Git**: Source control
- **GitHub**: Repository hosting
- Repository: `laxmikhilnani20/Finding_Missing_Person`
- Branch: `main`

## Documentation
- **README.md**: Comprehensive project documentation
- **DOCKER_SETUP.md**: Deployment guide
- **QUICK_REFERENCE.md**: Quick commands reference
- **Inline comments**: Code documentation

## Testing Approach
- Manual testing via web interface
- Camera connection testing built into UI
- No formal unit test framework (could add pytest)
- Integration testing via Docker deployment

## Monitoring & Logging
- **Console logging**: Print statements for debugging
- **Detection log**: CSV file with all detections
- **Error handling**: Try-catch blocks with error messages
- **Status indicators**: FPS, connection status, detection count

## Security Considerations
- No authentication implemented (add for production)
- CORS enabled for Socket.IO (`cors_allowed_origins="*"`)
- File uploads validated (image format only)
- URL validation for camera addresses
- No SQL injection risk (no database)

## Future Technical Improvements
- Add PostgreSQL/MongoDB for scalable storage
- Implement JWT authentication
- Add Redis for distributed state management
- Enable GPU acceleration in Docker (nvidia-docker)
- Add Prometheus metrics for monitoring
- Implement automated testing (pytest, Selenium)
- Add API rate limiting
- Enable HTTPS/TLS encryption
