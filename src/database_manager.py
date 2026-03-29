"""
Database Manager
Handles storage and retrieval of missing persons and detections
"""

import os
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import cv2


class DatabaseManager:
    """Manage missing persons database and detection logs"""
    
    def __init__(self, base_path="data"):
        """
        Initialize database manager
        
        Args:
            base_path (str): Base directory for data storage
        """
        self.base_path = base_path
        self.missing_persons_path = os.path.join(base_path, "missing_persons")
        self.detections_path = os.path.join(base_path, "detections")
        self.config_path = "config"
        
        # Create directories if they don't exist
        os.makedirs(self.missing_persons_path, exist_ok=True)
        os.makedirs(self.detections_path, exist_ok=True)
        os.makedirs(self.config_path, exist_ok=True)
        
        # Detection log
        self.detection_log_file = os.path.join(base_path, "detection_log.csv")
        self._initialize_detection_log()
    
    def _initialize_detection_log(self):
        """Initialize detection log CSV if it doesn't exist, or migrate if missing columns"""
        if not os.path.exists(self.detection_log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'person_name', 'camera_id', 'camera_name',
                'similarity', 'frame_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
            ])
            df.to_csv(self.detection_log_file, index=False)
        else:
            # Check if old format and upgrade if needed
            df = pd.read_csv(self.detection_log_file)
            required_cols = ['timestamp', 'person_name', 'camera_id', 'camera_name',
                           'similarity', 'frame_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
            
            # Add missing columns with None values
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
                    print(f"ℹ️ Added missing column: {col}")
            
            # Save upgraded CSV
            df[required_cols].to_csv(self.detection_log_file, index=False)
            print("✅ Detection log schema updated")
    
    def add_missing_person(self, name, image_file):
        """
        Add a missing person to the database
        
        Args:
            name (str): Person's name
            image_file: Uploaded image file
            
        Returns:
            str: Path to saved image or None if failed
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.missing_persons_path, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(person_dir, "profile.jpg")
            
            # Handle different image input types
            if hasattr(image_file, 'read'):
                # Streamlit UploadedFile
                img = Image.open(image_file)
                img = img.convert('RGB')
                img.save(image_path)
            elif isinstance(image_file, Image.Image):
                # PIL Image
                image_file.save(image_path)
            else:
                # File path
                img = Image.open(image_file)
                img = img.convert('RGB')
                img.save(image_path)
            
            print(f"✅ Added missing person: {name}")
            return image_path
            
        except Exception as e:
            print(f"❌ Error adding missing person {name}: {e}")
            return None
    
    def remove_missing_person(self, name):
        """Remove a missing person from database"""
        person_dir = os.path.join(self.missing_persons_path, name)
        
        if os.path.exists(person_dir):
            import shutil
            shutil.rmtree(person_dir)
            print(f"🗑️ Removed missing person: {name}")
            return True
        return False
    
    def get_missing_persons(self):
        """
        Get list of all missing persons
        
        Returns:
            dict: {name: image_path}
        """
        persons = {}
        
        if not os.path.exists(self.missing_persons_path):
            return persons
        
        for name in os.listdir(self.missing_persons_path):
            person_dir = os.path.join(self.missing_persons_path, name)
            if os.path.isdir(person_dir):
                image_path = os.path.join(person_dir, "profile.jpg")
                if os.path.exists(image_path):
                    persons[name] = image_path
        
        return persons
    
    def log_detection(self, person_name, camera_id, camera_name, similarity, frame, bbox=None):
        """
        Log a detection event
        
        Args:
            person_name (str): Name of detected person
            camera_id (str): Camera identifier
            camera_name (str): Camera name
            similarity (float): Match confidence
            frame: Frame image (numpy array)
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            str: Path to saved frame
        """
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:19]
            
            # Save frame
            frame_filename = f"{person_name}_{camera_id}_{timestamp_str}.jpg"
            frame_path = os.path.join(self.detections_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Extract bbox coordinates if provided
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = None, None, None, None
            if bbox and len(bbox) >= 4:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Add to log
            log_entry = {
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'person_name': person_name,
                'camera_id': camera_id,
                'camera_name': camera_name,
                'similarity': similarity,
                'frame_path': frame_path,
                'bbox_x1': bbox_x1,
                'bbox_y1': bbox_y1,
                'bbox_x2': bbox_x2,
                'bbox_y2': bbox_y2
            }
            
            df = pd.read_csv(self.detection_log_file)
            new_row = pd.DataFrame([log_entry])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.detection_log_file, index=False)
            
            print(f"📝 Logged detection: {person_name} at {camera_name}")
            return frame_path
            
        except Exception as e:
            print(f"❌ Error logging detection: {e}")
            return None
    
    def get_detection_log(self, limit=None):
        """
        Get detection log
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            pandas.DataFrame: Detection log
        """
        try:
            df = pd.read_csv(self.detection_log_file)
            if limit:
                df = df.tail(limit)
            return df
        except Exception as e:
            print(f"❌ Error reading detection log: {e}")
            return pd.DataFrame()
    
    def get_detections_for_person(self, person_name):
        """
        Get all detections for a specific person
        
        Args:
            person_name (str): Name of the person
            
        Returns:
            pandas.DataFrame: Detections for that person
        """
        try:
            df = pd.read_csv(self.detection_log_file)
            return df[df['person_name'] == person_name].sort_values('timestamp', ascending=False)
        except Exception as e:
            print(f"❌ Error getting detections for {person_name}: {e}")
            return pd.DataFrame()
    
    def export_detection_report(self, output_path=None):
        """
        Export detection report to CSV
        
        Args:
            output_path (str): Output file path
            
        Returns:
            str: Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.base_path, f"detection_report_{timestamp}.csv")
        
        try:
            df = pd.read_csv(self.detection_log_file)
            df.to_csv(output_path, index=False)
            print(f"📊 Exported detection report to {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return None
    
    def save_camera_config(self, cameras):
        """
        Save camera configuration
        
        Args:
            cameras (list): List of camera dictionaries
        """
        config_file = os.path.join(self.config_path, "cameras.json")
        try:
            with open(config_file, 'w') as f:
                json.dump(cameras, f, indent=2)
            print(f"💾 Saved camera configuration")
        except Exception as e:
            print(f"❌ Error saving camera config: {e}")
    
    def load_camera_config(self):
        """
        Load camera configuration
        
        Returns:
            list: List of camera dictionaries
        """
        config_file = os.path.join(self.config_path, "cameras.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    cameras = json.load(f)
                print(f"📂 Loaded camera configuration")
                return cameras
            return []
        except Exception as e:
            print(f"❌ Error loading camera config: {e}")
            return []
