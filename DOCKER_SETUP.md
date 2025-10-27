# 🐋 Docker Setup Guide - CCTV Missing Person Detection System

## Quick Start (Zero Local Dependencies!)

### Prerequisites
- **Docker** installed on your system
- **Docker Compose** (usually comes with Docker Desktop)
- That's it! No Python, OpenCV, or other dependencies needed locally.

---

## 🚀 Installation & Deployment

### Step 1: Clone the Repository
```bash
git clone https://github.com/laxmikhilnani20/Finding_Missing_Person.git
cd Finding_Missing_Person
```

### Step 2: Build the Docker Image
```bash
docker-compose build
```
This will:
- Create a containerized environment
- Install all Python dependencies
- Set up OpenCV and face recognition models
- Configure the Streamlit application

**Time:** First build takes ~5-10 minutes depending on your internet speed.

### Step 3: Run the Application
```bash
docker-compose up
```

The application will be available at:
```
http://localhost:8501
```

NOTE: The container image has been updated to run the Flask-based web UI (`flask_app.py`) via `gunicorn` instead of the previous Streamlit server. The exposed port (8501) remains the same, so the steps above are unchanged for development and deployment.

### Step 4: Stop the Application
Press `Ctrl+C` or run:
```bash
docker-compose down
```

---

## 📱 Setting Up IP Webcams

### Option 1: Using Android Phone as IP Camera

1. **Install IP Webcam App**
   - Download "IP Webcam" from Google Play Store
   - Open the app

2. **Start the Server**
   - Scroll to bottom → Click "Start Server"
   - Note the IP address shown (e.g., `http://192.168.1.100:8080`)

3. **Get the Video Stream URL**
   - The video stream URL is: `http://192.168.1.100:8080/video`
   - Use this URL in the application

### Option 2: Using iPhone as IP Camera

1. **Install EpocCam or iVCam**
   - Download from App Store
   - Follow app instructions to get stream URL

### Option 3: Using Laptop Webcam

1. **Use OBS Studio or VLC**
   - Stream your webcam over HTTP
   - Get the stream URL

### Option 4: Real IP Cameras

If you have actual IP cameras:
```
RTSP: rtsp://username:password@192.168.1.100:554/stream
HTTP: http://192.168.1.100/mjpeg
```

---

## 🎯 Using the System

### 1. Add IP Cameras

1. Open the app at `http://localhost:8501`
2. In the left sidebar → **Camera Management**
3. Click **"Add New Camera"**
4. Enter:
   - **Camera Name:** e.g., "Entrance"
   - **IP Camera URL:** e.g., `http://192.168.1.100:8080/video`
5. Click **"Test Connection"** to verify
6. Click **"Add Camera"**

### 2. Register Missing Persons

1. In sidebar → **Missing Persons**
2. Click **"Add Missing Person"**
3. Enter:
   - **Person Name:** e.g., "John Doe"
   - **Upload Photo:** Clear face photo
4. Click **"Add Person"**

### 3. Start Monitoring

1. Click **"▶️ Start Monitoring"** button
2. System will:
   - Connect to all cameras
   - Start face detection
   - Compare detected faces with registered persons
   - Alert when match is found

### 4. View Detections

- Switch to **"Detection Log"** tab
- View all detections with timestamps
- See detection images
- Export reports

---

## 🔧 Advanced Configuration

### Environment Variables

You can customize the application by editing `docker-compose.yml`:

```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501  # Change port
  - PYTHONUNBUFFERED=1
```

### Data Persistence

Data is stored in mounted volumes:
```
./data/missing_persons/  # Registered persons
./data/detections/       # Detection snapshots
./config/cameras.json    # Camera configurations
```

These persist even when container is stopped.

### Accessing Logs

View application logs:
```bash
docker-compose logs -f
```

---

## 🐛 Troubleshooting

### Cannot Connect to IP Camera

**Problem:** Camera connection fails

**Solutions:**
1. Ensure phone/camera and computer are on the **same WiFi network**
2. Check the IP address hasn't changed
3. Disable firewall temporarily
4. Test URL in browser first: `http://192.168.1.100:8080`

### Port Already in Use

**Problem:** Port 8501 is already in use

**Solution:** Change port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use 8502 instead
```

Then access at `http://localhost:8502`

### Slow Performance

**Problem:** System is slow

**Solutions:**
1. Reduce number of cameras
2. Lower frame rate by adjusting camera settings
3. Increase Docker memory allocation:
   - Docker Desktop → Settings → Resources → Memory (increase to 4GB+)

### No Face Detected

**Problem:** System doesn't detect faces

**Solutions:**
1. Ensure good lighting in camera view
2. Lower the **Confidence Threshold** (0.55-0.60)
3. Check uploaded photo has clear, front-facing face
4. Try different query image with better quality

---

## 📊 System Requirements

### Minimum:
- **RAM:** 4GB
- **CPU:** Dual-core
- **Storage:** 2GB free space
- **Network:** Local WiFi for IP cameras

### Recommended:
- **RAM:** 8GB+
- **CPU:** Quad-core
- **GPU:** CUDA-compatible (optional, for faster processing)
- **Storage:** 5GB+ free space

---

## 🛠️ Development Mode

### Run Without Docker (Not Recommended)

If you need to develop locally:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Rebuild After Changes

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## 📦 Deployment Options

### Deploy to Cloud (AWS/GCP/Azure)

1. Push Docker image to registry:
```bash
docker tag finding-missing-person:latest YOUR_REGISTRY/finding-missing-person:latest
docker push YOUR_REGISTRY/finding-missing-person:latest
```

2. Deploy on cloud VM with Docker installed
3. Ensure proper network configuration for IP camera access

### Deploy to Raspberry Pi

```bash
# On Raspberry Pi
docker-compose up -d
```

Access from any device on the network: `http://RASPBERRY_PI_IP:8501`

---

## 🎓 For PBL Presentation

### Demo Setup:

1. **Before Presentation:**
   - Start Docker container: `docker-compose up`
   - Add 2-3 phone cameras
   - Register 1-2 test persons
   - Test the system

2. **During Presentation:**
   - Show live camera feeds
   - Walk in front of camera
   - Demonstrate real-time detection
   - Show detection logs
   - Export and show report

3. **Highlight Docker Benefits:**
   - ✅ No dependency issues
   - ✅ Works on any platform
   - ✅ Easy to deploy
   - ✅ Production-ready

---

## 📧 Support

For issues or questions:
- **Email:** laxmikhilnani04@gmail.com
- **GitHub:** [laxmikhilnani20](https://github.com/laxmikhilnani20)

---

## 📄 License

This project is developed for academic purposes (PBL coursework).
