"""
IP Camera Manager
Handles multiple IP camera streams for CCTV monitoring
"""

import cv2
import threading
import queue
import time
from datetime import datetime


class IPCamera:
    """Individual IP camera handler"""
    
    def __init__(self, camera_id, name, url):
        """
        Initialize IP camera
        
        Args:
            camera_id (str): Unique camera identifier
            name (str): Human-readable camera name
            url (str): Camera stream URL (HTTP/RTSP)
        """
        self.camera_id = camera_id
        self.name = name
        self.url = url
        self.is_active = False
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=4)
        self.thread = None
        self.last_frame_time = None
        self._last_frame = None
        self.fps = 0
        
        # Configure RTSP specific settings
        self.is_rtsp = url.lower().startswith('rtsp://')
        if self.is_rtsp:
            # Extra FFMPEG options for RTSP streams
            self.stream_options = {
                cv2.CAP_PROP_HW_ACCELERATION: cv2.VIDEO_ACCELERATION_ANY,  # Enable HW acceleration
                cv2.CAP_PROP_BUFFERSIZE: 1,                               # Minimal buffering
                cv2.CAP_PROP_FPS: 30                                      # Target FPS
            }
        
    def connect(self):
        """Connect to camera stream"""
        try:
            if self.is_rtsp:
                # For RTSP streams, configure using FFmpeg backend explicitly
                self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                
                # Apply RTSP-specific options
                for prop, value in self.stream_options.items():
                    self.capture.set(prop, value)
                
                # Additional RTSP tuning
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            else:
                # For non-RTSP streams (like MJPEG)
                self.capture = cv2.VideoCapture(self.url)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            if self.capture.isOpened():
                # Test actual frame reading with timeout
                start_time = time.time()
                timeout = 5  # 5 seconds timeout
                while time.time() - start_time < timeout:
                    ret, test_frame = self.capture.read()
                    if ret and test_frame is not None:
                        self.is_active = True
                        # Get actual stream properties
                        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
                        codec = int(self.capture.get(cv2.CAP_PROP_FOURCC))
                        codec_str = ''.join([chr((codec >> 8*i) & 0xFF) for i in range(4)])
                        
                        print(f"✅ Connected to {self.name} ({self.url})")
                        print(f"   Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        print(f"   FPS: {actual_fps:.1f}")
                        print(f"   Codec: {codec_str}")
                        return True
                    time.sleep(0.1)
                
                print(f"❌ Timeout reading from camera {self.name} ({self.url})")
                self.capture.release()
                return False
            else:
                print(f"❌ Failed to connect to {self.name} ({self.url})")
                return False
                
        except Exception as e:
            print(f"❌ Error connecting to {self.name}: {e}")
            if self.capture:
                self.capture.release()
            return False
    
    def start_capture(self):
        """Start capturing frames in separate thread"""
        if not self.is_active:
            print(f"⚠️ Camera {self.name} is not connected!")
            return
        
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        print(f"▶️ Started capturing from {self.name}")
    
    def _capture_frames(self):
        """Internal method to capture frames continuously"""
        frame_count = 0
        start_time = time.time()
        last_fps_update = time.time()
        reconnect_delay = 1.0  # Start with 1 second delay
        max_reconnect_delay = 5.0
        failed_reads = 0
        max_failed_reads = 5  # Maximum consecutive failed reads before reconnecting
        
        while self.is_active:
            try:
                if self.is_rtsp:
                    # For RTSP: Use grabbing strategy to minimize latency
                    if not self.capture.grab():
                        failed_reads += 1
                        if failed_reads >= max_failed_reads:
                            print(f"⚠️ Multiple failed frame grabs from {self.name} - attempting reconnect in {reconnect_delay}s")
                            time.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                            self.capture.release()
                            if self.connect():
                                reconnect_delay = 1.0
                                failed_reads = 0
                            continue
                        time.sleep(0.1)
                        continue
                    
                    ret, frame = self.capture.retrieve()
                else:
                    # For non-RTSP: Direct read
                    ret, frame = self.capture.read()
                
                if not ret:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        print(f"⚠️ Multiple failed reads from {self.name} - attempting reconnect in {reconnect_delay}s")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                        self.capture.release()
                        if self.connect():
                            reconnect_delay = 1.0
                            failed_reads = 0
                        continue
                    time.sleep(0.1)
                    continue
                
                # Reset counters on successful frame
                failed_reads = 0
                reconnect_delay = 1.0
                
                # Update FPS every second
                frame_count += 1
                now = time.time()
                if now - last_fps_update >= 1.0:
                    self.fps = frame_count / (now - start_time)
                    last_fps_update = now
                
                # Clear old frame and add new one (non-blocking)
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                try:
                    self.frame_queue.put_nowait(frame)
                    self._last_frame = frame
                    self.last_frame_time = datetime.now()
                except queue.Full:
                    # If queue is full, keep the most recent frame in _last_frame
                    self._last_frame = frame
                    continue
                
            except Exception as e:
                print(f"❌ Error capturing from {self.name}: {e}")
                time.sleep(1)
    
    def get_frame(self):
        """Get latest frame from camera"""
        # Prefer returning the newest frame without blocking. If queue
        # is empty, fall back to the last seen frame (reduces blocking
        # latency and prevents generator stalls).
        try:
            frame = self.frame_queue.get_nowait()
            self._last_frame = frame
            return frame
        except queue.Empty:
            return self._last_frame
    
    def stop(self):
        """Stop camera capture"""
        self.is_active = False
        if self.capture:
            self.capture.release()
        print(f"⏹️ Stopped {self.name}")
    
    def get_status(self):
        """Get camera status"""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'url': self.url,
            'is_active': self.is_active,
            'fps': round(self.fps, 1),
            'last_frame': self.last_frame_time
        }


class IPCameraManager:
    """Manage multiple IP cameras"""
    
    def __init__(self):
        """Initialize camera manager"""
        self.cameras = {}
        self.monitoring = False
        
    def add_camera(self, camera_id, name, url):
        """
        Add new camera to system
        
        Args:
            camera_id (str): Unique identifier
            name (str): Camera name
            url (str): Stream URL
            
        Returns:
            bool: Success status
        """
        if camera_id in self.cameras:
            print(f"⚠️ Camera {camera_id} already exists!")
            return False
        
        camera = IPCamera(camera_id, name, url)
        if camera.connect():
            self.cameras[camera_id] = camera
            return True
        return False
    
    def remove_camera(self, camera_id):
        """Remove camera from system"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            print(f"🗑️ Removed camera {camera_id}")
            return True
        return False
    
    def start_all(self):
        """Start capturing from all cameras"""
        if not self.cameras:
            print("⚠️ No cameras added!")
            return False
        
        for camera in self.cameras.values():
            if camera.is_active:
                camera.start_capture()
        
        self.monitoring = True
        print(f"▶️ Started monitoring {len(self.cameras)} camera(s)")
        return True
    
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras.values():
            camera.stop()
        
        self.monitoring = False
        print("⏹️ Stopped all cameras")
    
    def get_frame(self, camera_id):
        """Get frame from specific camera"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame()
        return None
    
    def get_all_frames(self):
        """Get frames from all active cameras"""
        frames = {}
        for camera_id, camera in self.cameras.items():
            if camera.is_active:
                frame = camera.get_frame()
                if frame is not None:
                    frames[camera_id] = frame
        return frames
    
    def get_camera_info(self, camera_id):
        """Get camera information"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_status()
        return None
    
    def get_all_camera_info(self):
        """Get information for all cameras"""
        return {
            camera_id: camera.get_status()
            for camera_id, camera in self.cameras.items()
        }
    
    def test_connection(self, url):
        """
        Test if camera URL is accessible
        
        Args:
            url (str): Camera stream URL
            
        Returns:
            bool: Connection successful
        """
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
