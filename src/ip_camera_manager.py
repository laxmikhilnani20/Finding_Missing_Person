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
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.last_frame_time = None
        self.fps = 0
        
    def connect(self):
        """Connect to camera stream"""
        try:
            # Handle webcam device index (convert string to int)
            if isinstance(self.url, str) and self.url.isdigit():
                device_index = int(self.url)
                self.capture = cv2.VideoCapture(device_index)
            else:
                self.capture = cv2.VideoCapture(self.url)
            
            # Set buffer size to 1 for lower latency and disable buffering
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Request 60 FPS for maximum frame rate
                self.capture.set(cv2.CAP_PROP_FPS, 60)
                self.is_active = True
                print(f"✅ Connected to {self.name} ({self.url})")
                return True
            else:
                print(f"❌ Failed to connect to {self.name} ({self.url})")
                return False
        except Exception as e:
            print(f"❌ Error connecting to {self.name}: {e}")
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
        
        while self.is_active:
            try:
                # Check if capture is still valid
                if not self.capture or not self.capture.isOpened():
                    print(f"⚠️ Capture device closed for {self.name}, exiting thread")
                    break
                
                # Read frame directly without skipping for maximum FPS
                ret, frame = self.capture.read()
                
                if not ret:
                    print(f"⚠️ Failed to read frame from {self.name}")
                    time.sleep(0.01)
                    continue
                
                # Update FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    self.fps = frame_count / elapsed
                
                # Always replace old frame with newest one - keep only latest
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self.frame_queue.put(frame)
                self.last_frame_time = datetime.now()
                
            except Exception as e:
                print(f"❌ Error capturing from {self.name}: {e}")
                time.sleep(0.1)
                time.sleep(1)
    
    def get_frame(self):
        """Get latest frame from camera (non-blocking, always returns newest)"""
        try:
            # Get the most recent frame without blocking
            frame = None
            # Drain queue to get only the latest frame
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            return frame
        except Exception as e:
            print(f"❌ Error getting frame from {self.name}: {e}")
            return None
    
    def stop(self):
        """Stop camera capture"""
        print(f"⏹️ Stopping {self.name}...")
        self.is_active = False
        
        # Wait for thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Release capture after thread has stopped
        if self.capture:
            try:
                self.capture.release()
            except Exception as e:
                print(f"⚠️ Error releasing capture for {self.name}: {e}")
        
        print(f"✅ {self.name} stopped successfully")
    
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
            url (str): Camera stream URL or device index
            
        Returns:
            bool: Connection successful
        """
        try:
            # Handle webcam device index
            if isinstance(url, str) and url.isdigit():
                device_index = int(url)
                cap = cv2.VideoCapture(device_index)
            else:
                cap = cv2.VideoCapture(url)
            
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
