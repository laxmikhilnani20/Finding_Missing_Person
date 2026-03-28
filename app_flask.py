"""
CCTV-Based Missing Person Detection System - Flask Version
Real-time face recognition across multiple IP camera streams
"""

from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
from datetime import datetime
from PIL import Image
import pandas as pd
import io
import base64
import os

# Import custom modules
from src.face_recognition_engine import FaceRecognitionEngine
from src.ip_camera_manager import IPCameraManager
from src.database_manager import DatabaseManager
from src.utils import resize_frame, add_timestamp_overlay, add_alert_banner, validate_ip_url

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class AppState:
    def __init__(self):
        self.initialized = False
        self.face_engine = None
        self.camera_manager = None
        self.db_manager = DatabaseManager()
        self.monitoring = False
        self.query_embeddings = {}
        self.detection_count = 0
        self.last_detection = None
        self.monitoring_threads = {}

app_state = AppState()

def initialize_system():
    """Initialize face recognition engine and camera manager"""
    if not app_state.initialized:
        print("🚀 Initializing Face Recognition Engine...")
        app_state.face_engine = FaceRecognitionEngine(similarity_threshold=0.65)
        app_state.camera_manager = IPCameraManager()
        app_state.initialized = True
        
        # Load saved camera configuration
        saved_cameras = app_state.db_manager.load_camera_config()
        for cam in saved_cameras:
            app_state.camera_manager.add_camera(
                cam['id'], cam['name'], cam['url']
            )
        
        print("✅ System initialized successfully!")

def load_missing_persons():
    """Load missing persons and generate embeddings"""
    persons = app_state.db_manager.get_missing_persons()
    app_state.query_embeddings = {}
    
    for name, image_path in persons.items():
        try:
            image = Image.open(image_path)
            embedding = app_state.face_engine.encode_face(image)
            if embedding is not None:
                app_state.query_embeddings[name] = embedding
        except Exception as e:
            print(f"⚠️ Could not load embedding for {name}: {e}")

def process_camera_stream(camera_id):
    """Process individual camera stream and emit updates"""
    while app_state.monitoring:
        try:
            frame = app_state.camera_manager.get_frame(camera_id)
            
            if frame is not None:
                # Detect faces
                matches = app_state.face_engine.detect_and_match(
                    frame,
                    app_state.query_embeddings
                )
                
                # Draw matches
                if matches:
                    frame = app_state.face_engine.draw_matches(frame, matches)
                    frame = add_alert_banner(
                        frame,
                        f"🚨 {matches[0]['person_name'].upper()} DETECTED!"
                    )
                    
                    # Log detection
                    cam_info = app_state.camera_manager.get_camera_info(camera_id)
                    for match in matches:
                        app_state.db_manager.log_detection(
                            match['person_name'],
                            camera_id,
                            cam_info['name'],
                            match['similarity'],
                            frame
                        )
                        app_state.detection_count += 1
                        app_state.last_detection = datetime.now()
                        
                        # Emit detection alert
                        socketio.emit('detection_alert', {
                            'person_name': match['person_name'],
                            'camera_name': cam_info['name'],
                            'similarity': match['similarity'],
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                
                # Add timestamp overlay
                cam_info = app_state.camera_manager.get_camera_info(camera_id)
                frame = add_timestamp_overlay(frame, cam_info['name'])
                
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Emit frame update
                socketio.emit('frame_update', {
                    'camera_id': camera_id,
                    'frame': frame_base64,
                    'has_detection': len(matches) > 0
                })
            
            time.sleep(0.1)  # Adjust for desired FPS
            
        except Exception as e:
            print(f"❌ Error processing camera {camera_id}: {e}")
            time.sleep(1)

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    initialize_system()
    load_missing_persons()
    return render_template('index.html')

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras"""
    if app_state.camera_manager:
        cameras = app_state.camera_manager.get_all_camera_info()
        return jsonify({
            'success': True,
            'cameras': cameras
        })
    return jsonify({'success': False, 'message': 'System not initialized'})

@app.route('/api/cameras/add', methods=['POST'])
def add_camera():
    """Add new camera"""
    data = request.json
    cam_name = data.get('name')
    cam_url = data.get('url')
    
    if not cam_name or not cam_url:
        return jsonify({'success': False, 'message': 'Name and URL required'})
    
    if not validate_ip_url(cam_url):
        return jsonify({'success': False, 'message': 'Invalid URL format'})
    
    cam_id = f"cam_{int(time.time())}"
    if app_state.camera_manager.add_camera(cam_id, cam_name, cam_url):
        # Save configuration
        cameras = app_state.camera_manager.get_all_camera_info()
        camera_list = [
            {'id': cid, 'name': info['name'], 'url': info['url']}
            for cid, info in cameras.items()
        ]
        app_state.db_manager.save_camera_config(camera_list)
        
        return jsonify({
            'success': True,
            'message': f'Added {cam_name}',
            'camera_id': cam_id
        })
    
    return jsonify({'success': False, 'message': 'Failed to add camera'})

@app.route('/api/cameras/test', methods=['POST'])
def test_camera():
    """Test camera connection"""
    data = request.json
    cam_url = data.get('url')
    
    if not cam_url:
        return jsonify({'success': False, 'message': 'URL required'})
    
    if app_state.camera_manager.test_connection(cam_url):
        return jsonify({'success': True, 'message': 'Connection successful'})
    
    return jsonify({'success': False, 'message': 'Connection failed'})

@app.route('/api/cameras/remove/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    """Remove camera"""
    if app_state.camera_manager.remove_camera(camera_id):
        # Update saved configuration
        cameras = app_state.camera_manager.get_all_camera_info()
        camera_list = [
            {'id': cid, 'name': info['name'], 'url': info['url']}
            for cid, info in cameras.items()
        ]
        app_state.db_manager.save_camera_config(camera_list)
        
        return jsonify({'success': True, 'message': 'Camera removed'})
    
    return jsonify({'success': False, 'message': 'Camera not found'})

@app.route('/api/persons', methods=['GET'])
def get_persons():
    """Get all registered persons"""
    persons = app_state.db_manager.get_missing_persons()
    persons_list = []
    
    for name, image_path in persons.items():
        # Convert image to base64
        try:
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                persons_list.append({
                    'name': name,
                    'image': img_data
                })
        except:
            persons_list.append({
                'name': name,
                'image': None
            })
    
    return jsonify({
        'success': True,
        'persons': persons_list
    })

@app.route('/api/persons/add', methods=['POST'])
def add_person():
    """Add new missing person"""
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'success': False, 'message': 'Name and image required'})
    
    person_name = request.form['name']
    person_image = request.files['image']
    
    image_path = app_state.db_manager.add_missing_person(person_name, person_image)
    if image_path:
        load_missing_persons()
        return jsonify({
            'success': True,
            'message': f'Added {person_name}'
        })
    
    return jsonify({'success': False, 'message': 'Failed to add person'})

@app.route('/api/persons/remove/<person_name>', methods=['DELETE'])
def remove_person(person_name):
    """Remove missing person"""
    if app_state.db_manager.remove_missing_person(person_name):
        load_missing_persons()
        return jsonify({'success': True, 'message': f'Removed {person_name}'})
    
    return jsonify({'success': False, 'message': 'Person not found'})

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """Start monitoring all cameras"""
    if not app_state.query_embeddings:
        return jsonify({'success': False, 'message': 'Please add missing persons first'})
    
    if not app_state.camera_manager.cameras:
        return jsonify({'success': False, 'message': 'Please add cameras first'})
    
    app_state.camera_manager.start_all()
    app_state.monitoring = True
    
    # Start processing threads for each camera
    for camera_id in app_state.camera_manager.cameras.keys():
        thread = threading.Thread(target=process_camera_stream, args=(camera_id,), daemon=True)
        thread.start()
        app_state.monitoring_threads[camera_id] = thread
    
    return jsonify({'success': True, 'message': 'Monitoring started'})

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    app_state.camera_manager.stop_all()
    app_state.monitoring = False
    app_state.monitoring_threads.clear()
    
    return jsonify({'success': True, 'message': 'Monitoring stopped'})

@app.route('/api/monitoring/status', methods=['GET'])
def get_monitoring_status():
    """Get monitoring status"""
    return jsonify({
        'success': True,
        'monitoring': app_state.monitoring,
        'detection_count': app_state.detection_count
    })

@app.route('/api/threshold/update', methods=['POST'])
def update_threshold():
    """Update similarity threshold"""
    data = request.json
    threshold = data.get('threshold', 0.65)
    
    if app_state.face_engine:
        app_state.face_engine.set_similarity_threshold(threshold)
        return jsonify({'success': True, 'message': f'Threshold updated to {threshold}'})
    
    return jsonify({'success': False, 'message': 'System not initialized'})

@app.route('/api/detections/log', methods=['GET'])
def get_detection_log():
    """Get detection log"""
    limit = request.args.get('limit', 50, type=int)
    log_df = app_state.db_manager.get_detection_log(limit=limit)
    
    if not log_df.empty:
        log_data = log_df.to_dict('records')
        return jsonify({
            'success': True,
            'detections': log_data,
            'total': len(log_df),
            'unique_persons': log_df['person_name'].nunique(),
            'avg_confidence': log_df['similarity'].mean()
        })
    
    return jsonify({
        'success': True,
        'detections': [],
        'total': 0,
        'unique_persons': 0,
        'avg_confidence': 0
    })

@app.route('/api/detections/export', methods=['GET'])
def export_detections():
    """Export detection report"""
    report_path = app_state.db_manager.export_detection_report()
    
    if report_path:
        return send_file(
            report_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'detection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    return jsonify({'success': False, 'message': 'Failed to export report'})

@app.route('/api/detections/image/<path:filename>')
def get_detection_image(filename):
    """Get detection image"""
    try:
        return send_file(filename)
    except:
        return jsonify({'success': False, 'message': 'Image not found'}), 404

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_response', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
