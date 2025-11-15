# Quick Reference Guide

## 🚀 Quick Start Commands

### Start the Application
```bash
# Option 1: Using start script (recommended)
./start.sh

# Option 2: Using docker-compose directly
docker-compose up

# Option 3: Run in background
docker-compose up -d
```

### Stop the Application
```bash
# If running in foreground
Ctrl + C

# If running in background
docker-compose down
```

### Rebuild After Changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## 📱 IP Camera URLs

### Built-in Webcam
```
Built-in webcam: 0
External webcam 1: 1
External webcam 2: 2
```

### Smartphone as IP Camera

**Android (IP Webcam App):**
```
Video Stream: http://192.168.x.x:8080/video
Browser View: http://192.168.x.x:8080
```

**iOS (EpocCam):**
```
Follow app instructions for URL
```

### Real IP Cameras

**RTSP Stream:**
```
rtsp://username:password@192.168.1.100:554/stream1
rtsp://admin:admin123@192.168.1.100:554/h264
```

**HTTP/MJPEG:**
```
http://192.168.1.100/mjpeg
http://192.168.1.100:8080/video
```

---

## 🎯 Common Tasks

### Add a Camera
1. Sidebar → Camera Management → Add New Camera
2. Enter name: `Entrance`
3. Enter URL: `http://192.168.1.100:8080/video`
4. Test Connection
5. Add Camera

### Register Missing Person
1. Sidebar → Missing Persons → Add Missing Person
2. Enter name
3. Upload clear face photo
4. Add Person

### Start Detection
1. Click "▶️ Start Monitoring"
2. System begins real-time detection
3. Alerts appear on match

### Export Report
1. Click "📊 Export Report"
2. CSV file saved in `data/` folder

---

## 🔧 Troubleshooting

### Cannot Connect to Camera
```bash
# Test camera URL in browser first
http://192.168.1.100:8080

# Ensure devices on same WiFi
# Check firewall settings
# Verify IP address hasn't changed
```

### Port Already in Use
Edit `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Change to 8502
```

### Slow Performance
- Reduce number of cameras
- Increase Docker memory: Docker Desktop → Settings → Resources
- Lower camera resolution in camera app

### Docker Issues
```bash
# Restart Docker
docker-compose down
docker system prune -a  # Caution: removes all unused containers

# Rebuild fresh
docker-compose build --no-cache
```

---

## 📊 System Access

| Component | URL | Notes |
|-----------|-----|-------|
| Dashboard | `http://localhost:8501` | Main interface |
| Docker Logs | `docker-compose logs -f` | View system logs |
| Data Folder | `./data/` | All stored data |
| Detection Log | `./data/detection_log.csv` | All detections |

---

## 🎓 PBL Demo Quick Setup

**Pre-Demo (5 minutes):**
```bash
# 1. Start system
./start.sh

# 2. On 2-3 phones, install "IP Webcam"
# 3. Start server, note URLs

# 4. Add cameras to system
# 5. Register test person
# 6. Test detection
```

**During Demo:**
1. Show interface (1 min)
2. Add camera live (1 min)
3. Register person (1 min)
4. Start monitoring (2 min)
5. Walk in front of camera (2 min)
6. Show detection log & export (1 min)

---

## 📁 Important Files

```
app.py                          # Main application
src/face_recognition_engine.py  # Face recognition logic
src/ip_camera_manager.py        # Camera management
src/database_manager.py         # Data persistence
Dockerfile                      # Container config
docker-compose.yml              # Service setup
requirements.txt                # Python packages
```

---

## 🆘 Support

**Email:** laxmikhilnani04@gmail.com  
**GitHub:** [laxmikhilnani20](https://github.com/laxmikhilnani20)

**Documentation:**
- Full guide: [DOCKER_SETUP.md](DOCKER_SETUP.md)
- Main readme: [README.md](README.md)
