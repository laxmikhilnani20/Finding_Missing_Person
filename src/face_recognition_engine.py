"""
Face Recognition Engine
Handles face detection, encoding, and matching using MTCNN + InceptionResnetV1
"""

import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F


class FaceRecognitionEngine:
    """Face detection and recognition engine using FaceNet"""
    
    def __init__(self, similarity_threshold=0.65):
        """
        Initialize face recognition engine
        
        Args:
            similarity_threshold (float): Minimum similarity score for match (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        print(f"🚀 Initializing Face Recognition Engine on {self.device}...")
        self.mtcnn = MTCNN(
            keep_all=True,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print("✅ Models loaded successfully!")
        
    def encode_face(self, image):
        """
        Encode a face from an image
        
        Args:
            image: PIL Image or numpy array (RGB)
            
        Returns:
            torch.Tensor: 512-d face embedding or None if no face detected
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Detect face
        boxes, _ = self.mtcnn.detect(image)
        
        if boxes is None:
            return None
        
        # Get face tensor
        face_tensor = self.mtcnn.forward(image, save_path=None)
        
        if face_tensor is None:
            return None
        
        # Take first face if multiple detected
        if len(face_tensor.shape) == 3:
            face_tensor = face_tensor.unsqueeze(0)
        
        face_tensor = face_tensor[0].unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
        
        return embedding
    
    def detect_and_match(self, frame, query_embeddings):
        """
        Detect faces in frame and match against query embeddings
        
        Args:
            frame: numpy array (BGR format from cv2)
            query_embeddings: dict of {person_name: embedding_tensor}
            
        Returns:
            list: Detected matches and unknown faces with format:
                  [{
                      'person_name': str (or 'Unknown Person'),
                      'bbox': [x1, y1, x2, y2],
                      'similarity': float,
                      'is_match': bool
                  }]
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Detect faces
        boxes, _ = self.mtcnn.detect(pil_image)
        
        if boxes is None:
            return []
        
        # Get face embeddings
        face_tensors = self.mtcnn.forward(pil_image, save_path=None)
        
        if face_tensors is None:
            return []
        
        face_tensors = face_tensors.to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            face_embeddings = self.resnet(face_tensors)
        
        matches = []
        
        # Ensure we have matching boxes and embeddings
        num_faces = min(len(boxes), len(face_embeddings))
        
        # Compare each detected face with all query embeddings
        for i in range(num_faces):
            box = boxes[i]
            face_embedding = face_embeddings[i]
            
            # Validate bbox format
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
                print(f"⚠️ Invalid bbox format: {box} (type: {type(box)}), skipping face {i}")
                continue
            
            # Safely convert bbox to list of ints
            try:
                bbox_list = [int(b) for b in box]
            except (TypeError, ValueError) as e:
                print(f"⚠️ Error converting bbox to int list: {box}, error: {e}")
                continue
            
            best_match = None
            best_similarity = 0.0
            
            for person_name, query_embedding in query_embeddings.items():
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    face_embedding.unsqueeze(0),
                    query_embedding
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
            
            # Add to results - mark as match or unknown
            if best_similarity >= self.similarity_threshold:
                matches.append({
                    'person_name': best_match,
                    'bbox': bbox_list,
                    'similarity': best_similarity,
                    'is_match': True
                })
            else:
                # Add as unknown person to show detection is working
                matches.append({
                    'person_name': 'Unknown Person',
                    'bbox': bbox_list,
                    'similarity': best_similarity,
                    'is_match': False
                })
        
        return matches
    
    def draw_matches(self, frame, matches):
        """
        Draw bounding boxes and labels on frame with color coding
        
        Args:
            frame: numpy array (BGR format)
            matches: list of match dictionaries
            
        Returns:
            numpy array: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for match in matches:
            # Validate bbox format before unpacking
            bbox = match.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                print(f"⚠️ Skipping invalid bbox in draw_matches: {bbox}")
                continue
            
            x1, y1, x2, y2 = bbox
            person_name = match['person_name']
            similarity = match['similarity']
            is_match = match.get('is_match', True)
            
            # Color coding: RED for known persons, YELLOW for unknown faces
            if is_match:
                box_color = (0, 0, 255)  # RED - Known person (BGR format)
                label_bg_color = (0, 0, 255)
                thickness = 3
                label = f"✓ {person_name}: {similarity:.2%}"
            else:
                box_color = (0, 255, 255)  # YELLOW - Unknown person (BGR format)
                label_bg_color = (0, 200, 200)
                thickness = 2
                label = f"? Unknown: {similarity:.2%}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                label_bg_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def set_similarity_threshold(self, threshold):
        """Update similarity threshold"""
        self.similarity_threshold = threshold
        print(f"🎯 Similarity threshold updated to {threshold}")
