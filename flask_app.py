from flask import Flask, render_template, Response, request, redirect, url_for, flash
import time
import cv2
try:
    # Try to import the real face engine (may fail if facenet_pytorch isn't installed)
    from src.face_recognition_engine import FaceRecognitionEngine
    FACE_ENGINE_AVAILABLE = True
except Exception as _e:
    print("⚠️ facenet_pytorch or model deps not available — running in streaming-only fallback mode")
    FACE_ENGINE_AVAILABLE = False

from src.ip_camera_manager import IPCameraManager
from src.database_manager import DatabaseManager
from src.utils import resize_frame, add_timestamp_overlay, add_alert_banner, validate_ip_url
from PIL import Image
from queue import Queue
import json


class DummyFaceRecognitionEngine:
    """Fallback engine used when facenet_pytorch isn't installed.

    This implements the minimal API used by the Flask app so streaming works
    without ML features (encode_face -> None, detect_and_match -> []).
    """
    def __init__(self, similarity_threshold=0.65):
        self.similarity_threshold = similarity_threshold

    def encode_face(self, image):
        return None

    def detect_and_match(self, frame, query_embeddings):
        return []

    def draw_matches(self, frame, matches):
        return frame

    def set_similarity_threshold(self, threshold):
        self.similarity_threshold = threshold


app = Flask(__name__)
app.secret_key = "replace-with-secure-key"

# Initialize core components (reuse existing project classes)
db_manager = DatabaseManager()
if FACE_ENGINE_AVAILABLE:
    try:
        face_engine = FaceRecognitionEngine(similarity_threshold=0.65)
    except Exception as _e:
        print(f"⚠️ Failed to initialize FaceRecognitionEngine: {_e} — falling back to dummy engine")
        face_engine = DummyFaceRecognitionEngine()
else:
    face_engine = DummyFaceRecognitionEngine()

camera_manager = IPCameraManager()

# Simple SSE pub/sub: each connected client gets its own Queue
subscribers = []

def publish_event(event_dict):
    """Publish event (dict) to all connected SSE clients as JSON string."""
    data = json.dumps(event_dict)
    for q in list(subscribers):
        try:
            q.put(data, block=False)
        except Exception:
            # ignore full/closed queues
            pass


@app.route('/events')
def events():
    """SSE endpoint that streams JSON events to the browser."""
    def event_stream(q: Queue):
        try:
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        except GeneratorExit:
            # client disconnected
            return
        finally:
            try:
                subscribers.remove(q)
            except ValueError:
                pass

    q = Queue()
    subscribers.append(q)
    return Response(event_stream(q), mimetype='text/event-stream')

# Load saved cameras from config
for cam in db_manager.load_camera_config():
    camera_manager.add_camera(cam['id'], cam['name'], cam['url'])

# Load missing persons and build embeddings
query_embeddings = {}
def load_missing_persons_embeddings():
    global query_embeddings
    query_embeddings = {}
    persons = db_manager.get_missing_persons()
    for name, image_path in persons.items():
        try:
            img = Image.open(image_path)
            emb = face_engine.encode_face(img)
            if emb is not None:
                query_embeddings[name] = emb
        except Exception as e:
            print(f"❌ Failed to encode {name}: {e}")


load_missing_persons_embeddings()


def gen_mjpeg(camera_id):
    """Generator that yields MJPEG frames for a given camera id."""
    frame_interval = 1.0 / 25  # Target 25 FPS
    last_frame_time = 0
    detection_interval = 1.0  # Run detection every 1.0 seconds (reduces CPU)
    last_detection_time = 0
    
    while True:
        current_time = time.time()
        
        # Maintain consistent frame rate
        if current_time - last_frame_time < frame_interval:
            time.sleep(max(0, frame_interval - (current_time - last_frame_time)))
            continue
        
        frame = camera_manager.get_frame(camera_id)
        if frame is None:
            time.sleep(0.016)  # ~60fps sleep time
            continue
            
        try:
            # Run detection at intervals
            should_detect = current_time - last_detection_time >= detection_interval
            if should_detect:
                matches = face_engine.detect_and_match(frame, query_embeddings)
                if matches:
                    frame = face_engine.draw_matches(frame, matches)
                    frame = add_alert_banner(frame, f"🚨 {matches[0]['person_name'].upper()} DETECTED!")
                # Log detections and publish SSE events
                cam_info = camera_manager.get_camera_info(camera_id) or {}
                for match in matches:
                    db_manager.log_detection(
                        match['person_name'], camera_id, cam_info.get('name', camera_id), match['similarity'], frame
                    )
                    # Publish event to connected clients
                    event = {
                        'type': 'detection',
                        'camera_id': camera_id,
                        'camera_name': cam_info.get('name', camera_id),
                        'person_name': match['person_name'],
                        'similarity': match['similarity']
                    }
                    try:
                        publish_event(event)
                    except Exception:
                        pass

            # Add timestamp overlay and resize for streaming
            cam_info = camera_manager.get_camera_info(camera_id) or {}
            frame = add_timestamp_overlay(frame, cam_info.get('name', camera_id))
            frame = resize_frame(frame, max_width=800)
            # Encode to JPEG with reasonable quality to reduce CPU/network
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except GeneratorExit:
            break
        except Exception as e:
            print(f"❌ Error in mjpeg generator for {camera_id}: {e}")
            time.sleep(0.5)


@app.route('/')
def index():
    cameras = camera_manager.get_all_camera_info()
    persons = db_manager.get_missing_persons()
    detection_log = db_manager.get_detection_log(limit=1)
    detection_count = 0
    try:
        detection_count = len(db_manager.get_detection_log())
    except:
        detection_count = 0

    return render_template('index.html', cameras=cameras, persons=persons, monitoring=camera_manager.monitoring, detection_count=detection_count)


@app.route('/add_camera', methods=['POST'])
def add_camera():
    name = request.form.get('camera_name')
    url = request.form.get('camera_url')
    if not name or not url:
        flash('Please provide both camera name and URL', 'warning')
        return redirect(url_for('index'))

    if not validate_ip_url(url):
        flash('Invalid camera URL', 'danger')
        return redirect(url_for('index'))

    cam_id = f"cam_{int(time.time())}"
    if camera_manager.add_camera(cam_id, name, url):
        # Start capturing immediately if monitoring is active
        if camera_manager.monitoring:
            cam = camera_manager.cameras.get(cam_id)
            if cam:
                cam.start_capture()
        
        # Persist config
        cams = camera_manager.get_all_camera_info()
        camera_list = [{'id': cid, 'name': info['name'], 'url': info['url']} for cid, info in cams.items()]
        db_manager.save_camera_config(camera_list)
        flash(f'Added camera {name}', 'success')
    else:
        flash('Failed to add camera (connection failed)', 'danger')

    return redirect(url_for('index'))


@app.route('/remove_camera/<camera_id>')
def remove_camera(camera_id):
    camera_manager.remove_camera(camera_id)
    cams = camera_manager.get_all_camera_info()
    camera_list = [{'id': cid, 'name': info['name'], 'url': info['url']} for cid, info in cams.items()]
    db_manager.save_camera_config(camera_list)
    flash('Camera removed', 'info')
    return redirect(url_for('index'))


@app.route('/add_person', methods=['POST'])
def add_person():
    name = request.form.get('person_name')
    file = request.files.get('person_image')
    if not name or not file:
        flash('Please provide both name and image', 'warning')
        return redirect(url_for('index'))

    saved_path = db_manager.add_missing_person(name, file)
    if saved_path:
        load_missing_persons_embeddings()
        flash(f'Added person {name}', 'success')
    else:
        flash('Failed to add person', 'danger')

    return redirect(url_for('index'))


@app.route('/start')
def start_monitoring():
    if not query_embeddings:
        flash('Please add missing persons before starting monitoring', 'warning')
        return redirect(url_for('index'))

    if not camera_manager.cameras:
        flash('Please add cameras before starting monitoring', 'warning')
        return redirect(url_for('index'))

    camera_manager.start_all()
    flash('Monitoring started', 'success')
    return redirect(url_for('index'))


@app.route('/stop')
def stop_monitoring():
    camera_manager.stop_all()
    flash('Monitoring stopped', 'info')
    return redirect(url_for('index'))


@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Returns MJPEG stream for the given camera."""
    # Make sure camera capture is started
    if not camera_manager.monitoring:
        if camera_id in camera_manager.cameras:
            camera = camera_manager.cameras[camera_id]
            if not camera.is_active:
                camera.connect()
                camera.start_capture()
    
    # Return MJPEG stream
    return Response(
        gen_mjpeg(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    # Run Flask dev server on port 8501 (same as Streamlit default) for easy swap
    app.run(host='0.0.0.0', port=8501, threaded=True)
