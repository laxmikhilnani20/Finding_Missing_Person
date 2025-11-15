"""
Utility Functions
Helper functions for the application
"""

import cv2
import numpy as np
from datetime import datetime


def resize_frame(frame, max_width=640, max_height=480):
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: numpy array (image)
        max_width (int): Maximum width
        max_height (int): Maximum height
        
    Returns:
        numpy array: Resized frame
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame


def add_timestamp_overlay(frame, camera_name):
    """
    Add timestamp and camera name overlay to frame
    
    Args:
        frame: numpy array (image)
        camera_name (str): Camera name
        
    Returns:
        numpy array: Frame with overlay
    """
    overlay_frame = frame.copy()
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add semi-transparent background
    overlay = overlay_frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, overlay_frame, 0.5, 0, overlay_frame)
    
    # Add text
    cv2.putText(
        overlay_frame,
        camera_name,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    cv2.putText(
        overlay_frame,
        timestamp,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )
    
    return overlay_frame


def create_grid_layout(frames_dict, grid_cols=2):
    """
    Create grid layout for multiple camera feeds
    
    Args:
        frames_dict (dict): {camera_id: frame}
        grid_cols (int): Number of columns in grid
        
    Returns:
        numpy array: Combined grid image
    """
    if not frames_dict:
        return None
    
    # Resize all frames to same size
    target_size = (640, 480)
    resized_frames = []
    
    for camera_id, frame in frames_dict.items():
        resized = cv2.resize(frame, target_size)
        resized_frames.append(resized)
    
    # Calculate grid dimensions
    num_frames = len(resized_frames)
    grid_rows = (num_frames + grid_cols - 1) // grid_cols
    
    # Create empty frames for incomplete grid
    while len(resized_frames) < grid_rows * grid_cols:
        empty_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        resized_frames.append(empty_frame)
    
    # Combine frames into grid
    rows = []
    for i in range(grid_rows):
        start_idx = i * grid_cols
        end_idx = start_idx + grid_cols
        row_frames = resized_frames[start_idx:end_idx]
        row_combined = np.hstack(row_frames)
        rows.append(row_combined)
    
    grid = np.vstack(rows)
    return grid


def add_alert_banner(frame, message="🚨 PERSON DETECTED!", blink=True):
    """
    Add alert banner to frame
    
    Args:
        frame: numpy array (image)
        message (str): Alert message
        blink (bool): Whether to blink
        
    Returns:
        numpy array: Frame with alert banner
    """
    alert_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Blink effect (show banner every other second)
    if blink:
        second = datetime.now().second
        if second % 2 == 0:
            return alert_frame
    
    # Draw red banner
    cv2.rectangle(alert_frame, (0, height - 60), (width, height), (0, 0, 255), -1)
    
    # Add text (using FONT_HERSHEY_SIMPLEX which is reliable)
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 20
    
    cv2.putText(
        alert_frame,
        message,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    
    return alert_frame


def validate_ip_url(url):
    """
    Validate IP camera URL format or webcam device
    
    Args:
        url (str): Camera URL or device index/path
        
    Returns:
        bool: Whether URL is valid
    """
    url = url.strip()
    
    # Check if it's a webcam device index (0, 1, 2, etc.)
    if url.isdigit():
        return True
    
    # Check if it's a device path (/dev/video0, /dev/video1, etc.)
    if url.startswith('/dev/video'):
        return True
    
    # Check if URL starts with http://, https://, or rtsp://
    valid_protocols = ['http://', 'https://', 'rtsp://']
    
    if not any(url.startswith(protocol) for protocol in valid_protocols):
        return False
    
    # Basic validation
    if len(url) < 10:
        return False
    
    return True


def format_detection_stats(detections_count, cameras_active):
    """
    Format detection statistics
    
    Args:
        detections_count (int): Number of detections
        cameras_active (int): Number of active cameras
        
    Returns:
        str: Formatted statistics
    """
    return f"📊 Active Cameras: {cameras_active} | 🔍 Total Detections: {detections_count}"
