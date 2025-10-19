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
            list: Detected matches with format:
                  [{
                      'person_name': str,
                      'bbox': [x1, y1, x2, y2],
                      'similarity': float
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
        
        # Compare each detected face with all query embeddings
        for i, (box, face_embedding) in enumerate(zip(boxes, face_embeddings)):
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
            
            # Check if match exceeds threshold
            if best_similarity >= self.similarity_threshold:
                matches.append({
                    'person_name': best_match,
                    'bbox': [int(b) for b in box],
                    'similarity': best_similarity
                })
        
        return matches
    
    def draw_matches(self, frame, matches):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: numpy array (BGR format)
            matches: list of match dictionaries
            
        Returns:
            numpy array: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for match in matches:
            x1, y1, x2, y2 = match['bbox']
            person_name = match['person_name']
            similarity = match['similarity']
            
            # Draw bounding box (red for alert)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label background
            label = f"{person_name}: {similarity:.2%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 0, 255),
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def set_similarity_threshold(self, threshold):
        """Update similarity threshold"""
        self.similarity_threshold = threshold
        print(f"🎯 Similarity threshold updated to {threshold}")
