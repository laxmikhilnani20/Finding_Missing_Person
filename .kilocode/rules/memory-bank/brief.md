# Project Brief

## Project Name
CCTV-Based Missing Person Detection System

## Purpose
A production-ready real-time face recognition system designed to detect missing persons across multiple IP camera streams using deep learning. Built for PBL (Project-Based Learning) 2025.

## Core Requirements
1. Monitor multiple IP cameras simultaneously (HTTP, HTTPS, RTSP streams)
2. Detect and identify missing persons in real-time using facial recognition
3. Provide instant alerts when a match is found with confidence scores
4. Maintain comprehensive detection logs with timestamps and snapshots
5. Support adding/removing cameras and missing persons dynamically
6. Deploy easily with zero local dependencies using Docker

## Key Goals
- **Real-time Performance**: Process video streams with minimal latency
- **Accuracy**: Achieve >95% detection accuracy with adjustable confidence thresholds
- **Scalability**: Support multiple cameras (5-10 streams on 8GB RAM)
- **Usability**: Modern, intuitive web interface accessible via browser
- **Reliability**: 24/7 monitoring capability with proper error handling
- **Portability**: Complete containerization for deployment anywhere

## Success Criteria
- System can monitor multiple cameras simultaneously
- Face detection accuracy >95% in good lighting conditions
- Real-time WebSocket notifications with <1 second delay
- CSV export functionality for detection reports
- Persistent storage of camera configurations and detection logs
- Docker deployment working on any platform

## Project Scope
**In Scope:**
- Multi-camera video stream processing
- Face detection using MTCNN
- Face recognition using FaceNet (InceptionResnetV1)
- Web-based dashboard with live feeds
- Detection logging and reporting
- Camera and person management
- Docker containerization

**Out of Scope:**
- Mobile applications
- Cloud deployment (future enhancement)
- Database integration (file-based storage sufficient)
- Advanced analytics (heatmaps, patterns)
- Multi-modal recognition (clothing, gait)
- User authentication/authorization

## Target Users
- Healthcare facilities (dementia/Alzheimer's patients)
- Campus security (missing students)
- Corporate security (employee safety)
- Law enforcement (public safety)
- Event security (crowd monitoring)

## Project Evolution
- **Phase 1**: Jupyter notebook prototype for video analysis
- **Phase 2**: Streamlit app for interactive interface
- **Phase 3**: Flask + WebSocket production application (Current)
