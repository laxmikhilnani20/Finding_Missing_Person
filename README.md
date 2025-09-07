# Seek & Spot: CNN-Based Person Tracking in Videos

A CNN-based framework for automatic person search and detection across multiple video datasets using deep learning techniques.

## 📋 Project Overview

This project addresses the challenge of efficiently searching and identifying individuals from large volumes of video data. Using state-of-the-art deep learning models, we've developed an intelligent system that can automatically detect and track a target person across multiple videos with high accuracy and efficiency.

## 🎯 Problem Statement

Manual searching and identification of individuals from video sources is:
- Extremely time-consuming and inefficient
- Subject to human errors
- Impractical for large-scale video datasets
- Limited by conventional methods that work poorly with dynamic video content

## 🚀 Objectives

1. **Automated Person Detection**: Create a CNN-based framework for automatic person search across multiple video datasets
2. **Efficient Processing**: Implement batch processing for scalable performance
3. **Robust Pipeline**: Develop a comprehensive face recognition pipeline including:
   - Face detection from video frames
   - Feature embedding using pre-trained models
   - Similarity matching between query and dataset embeddings
4. **Structured Output**: Generate detailed reports with timestamps and frame indices
5. **User-Friendly Interface**: Deploy a Flask-based web application
6. **Future Enhancement**: Framework for temporal consistency improvements

## 🛠️ Technical Stack

### Core Technologies
- **Deep Learning Frameworks**: PyTorch/TensorFlow
- **Computer Vision**: OpenCV
- **Web Framework**: Flask
- **Data Processing**: NumPy, Pandas

### Models & Algorithms
- **Face Detection**: RetinaFace / MTCNN
- **Face Embedding**: ArcFace / FaceNet
- **Similarity Matching**: Cosine similarity / Euclidean distance

### Key Parameters
- **Input Image Size**: 112×112 pixels
- **Embedding Dimension**: 512
- **Batch Processing**: Optimized batch sizes
- **Output Format**: JSON/CSV reports with timestamps

## 📁 Project Structure

```
Finding_Missing_Person/
├── PBL.ipynb                    # Main notebook with implementation
├── face_match_model.pkl         # Trained model file
├── all_faces_report.csv         # Comprehensive face detection report
├── output_report.csv           # Final detection results
├── inputs/                     # Input files directory
│   ├── test_video.mp4          # Sample video file
│   └── query_image.png         # Sample query image
└── README.md                   # Project documentation
```

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/laxmikhilnani20/Finding_Missing_Person.git
   cd Finding_Missing_Person
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install opencv-python
   pip install flask
   pip install numpy pandas
   pip install face-recognition
   pip install mtcnn
   ```

3. **Download pre-trained models**
   - The project uses pre-trained ArcFace/FaceNet models
   - Models will be automatically downloaded on first run

## 🚀 Usage

### Running the Jupyter Notebook
1. Open `PBL.ipynb` in Jupyter Notebook or VS Code
2. Run all cells to process the sample data
3. Check the generated reports in CSV format

### Using the Flask Web Application
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Upload a query image and video files
3. Get detailed detection reports with timestamps

## 📊 Features

- **Real-time Processing**: Efficient batch processing for large video datasets
- **High Accuracy**: State-of-the-art CNN models for robust face recognition
- **Detailed Reports**: Comprehensive output with timestamps and frame indices
- **Scalable Architecture**: Designed to handle multiple videos simultaneously
- **User-Friendly Interface**: Web-based application for easy interaction

## 🔍 How It Works

1. **Face Detection**: Extract faces from video frames using RetinaFace/MTCNN
2. **Feature Extraction**: Generate embeddings using ArcFace/FaceNet
3. **Similarity Matching**: Compare query face with detected faces using cosine similarity
4. **Result Generation**: Create detailed reports with detection timestamps
5. **Output Delivery**: Present results in structured CSV/JSON format

## 📈 Performance Metrics

- **Accuracy**: High precision in face detection and matching
- **Efficiency**: Optimized for processing speed vs. accuracy balance
- **Scalability**: Handles multiple videos with batch processing
- **Reliability**: Robust performance across various lighting and pose conditions

## 🔮 Future Enhancements

- **Temporal Consistency**: Implementation of RNNs or Transformers for extended video sequences
- **Real-time Processing**: Live video stream analysis capabilities
- **Mobile Deployment**: Mobile application development
- **Advanced Tracking**: Integration of DeepSORT for improved tracking

## 📚 References

1. Schroff, F., et al. (2015). FaceNet: A unified embedding for face recognition and clustering. CVPR.
2. Deng, J., et al. (2019). ArcFace: Additive angular margin loss for deep face recognition. CVPR.
3. Zhang, K., et al. (2016). Joint face detection and alignment using multitask cascaded convolutional networks.
4. Deng, J., et al. (2019). RetinaFace: Single-stage dense face localisation in the wild.
5. OpenCV Documentation: https://opencv.org/
6. PyTorch Documentation: https://pytorch.org/
7. Flask Documentation: https://flask.palletsprojects.com/

## 📄 License

This project is developed as part of academic coursework. Please refer to the institution's guidelines for usage and distribution.

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can collaborate:

### How to Contribute

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub or use GitHub CLI
   gh repo fork laxmikhilnani20/Finding_Missing_Person
   ```

2. **Clone your forked repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Finding_Missing_Person.git
   cd Finding_Missing_Person
   ```

3. **Create a new branch for your feature**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes and commit**
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

5. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to your forked repository on GitHub
   - Click "New Pull Request"
   - Describe your changes and submit

### What You Can Contribute
- Bug fixes and improvements
- New features or enhancements
- Documentation improvements
- Performance optimizations
- Code refactoring

Feel free to open an issue first to discuss major changes!

## 📧 Contact

For any doubts, questions, or suggestions regarding this project, feel free to reach out:

- **Email**: [laxmikhilnani04@gmail.com](mailto:laxmikhilnani04@gmail.com)
- **GitHub**: [laxmikhilnani20](https://github.com/laxmikhilnani20)

---

**Note**: This project was developed as part of PBL (Project Based Learning) coursework in August 2025.