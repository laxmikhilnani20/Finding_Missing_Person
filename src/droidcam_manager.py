"""
DroidCam Manager
Handles multiple DroidCam smartphone streams for CCTV monitoring
DroidCam connects via IP address + port (e.g., http://192.168.x.x:4747/video)
"""

import cv2
import threading
import queue
import time
from datetime import datetime


class DroidCam:
    """Individual DroidCam handler"""
    
    def __init__(self, camera_id, name, ip_address, port=4747):
        """
        Initialize DroidCam
        
        Args:
            camera_id (str): Unique camera identifier
            name (str): Human-readable camera name
            ip_address (str): IP address of smartphone running DroidCam
            port (int): DroidCam port (default 4747)
        """
        self.camera_id = camera_id
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.url = f"http://{ip_address}:{port}/video"
        self.is_active = False
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.last_frame_time = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        
    def connect(self):
        """Connect to DroidCam stream"""
        try:
            self.capture = cv2.VideoCapture(self.url)
            # Set buffer size to 1 for lower latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.capture.isOpened():
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
        
        self.frame_count = 0
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        print(f"▶️ Started capturing from {self.name}")
    
    def _capture_frames(self):
        """Internal method to capture frames continuously"""
        while self.is_active:
            try:
                ret, frame = self.capture.read()
                
                if not ret:
                    print(f"⚠️ Failed to read frame from {self.name}, reconnecting...")
                    time.sleep(1)
                    self._reconnect()
                    continue
                
                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
                
                # Clear old frame and add new one (keep buffer small for low latency)
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                self.last_frame_time = datetime.now()
                
            except Exception as e:
                print(f"❌ Error capturing from {self.name}: {e}")
                time.sleep(1)
    
    def _reconnect(self):
        """Attempt to reconnect to DroidCam"""
        try:
            if self.capture:
                self.capture.release()
            time.sleep(2)
            self.capture = cv2.VideoCapture(self.url)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"❌ Reconnection failed for {self.name}: {e}")
    
    def get_frame(self):
        """Get latest frame from camera"""
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None
    
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
            'ip_address': self.ip_address,
            'port': self.port,
            'url': self.url,
            'is_active': self.is_active,
            'fps': round(self.fps, 1),
            'last_frame': self.last_frame_time
        }


class DroidCamManager:
    """Manage multiple DroidCam devices"""
    
    def __init__(self):
        """Initialize camera manager"""
        self.cameras = {}
        self.monitoring = False
        
    def add_camera(self, camera_id, name, ip_address, port=4747):
        """
        Add new DroidCam to system
        
        Args:
            camera_id (str): Unique identifier
            name (str): Camera name
            ip_address (str): IP address of smartphone
            port (int): DroidCam port (default 4747)
            
        Returns:
            bool: Success status
        """
        if camera_id in self.cameras:
            print(f"⚠️ Camera {camera_id} already exists!")
            return False
        
        camera = DroidCam(camera_id, name, ip_address, port)
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
        print(f"▶️ Started monitoring {len(self.cameras)} DroidCam(s)")
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
    
    def test_connection(self, ip_address, port=4747):
        """
        Test if DroidCam is accessible
        
        Args:
            ip_address (str): IP address of smartphone
            port (int): DroidCam port
            
        Returns:
            bool: Connection successful
        """
        try:
            url = f"http://{ip_address}:{port}/video"
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
