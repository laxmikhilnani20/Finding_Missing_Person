# DroidCam Setup Guide

This project now supports **DroidCam** - using your smartphone as a camera instead of IP cameras.

## What is DroidCam?

DroidCam is a mobile application that turns your smartphone into a wireless camera that streams video over WiFi. It's perfect for surveillance systems, and provides:
- **Low latency** video streaming
- **Easy setup** - no complex IP configuration
- **Better performance** for FPS and detection speed
- **Mobile flexibility** - place cameras anywhere

## Installation Steps

### 1. Install DroidCam App on Your Smartphone

**For Android:**
- Download from Google Play Store: https://play.google.com/store/apps/details?id=com.dev47apps.droidcam
- Or use the free version: DroidCam - Webcam

**For iOS:**
- Download from App Store: https://apps.apple.com/app/droidcam-webcam/id1510258102

### 2. Configure DroidCam

1. Open the DroidCam app on your phone
2. Make sure your phone is connected to the **same WiFi network** as your computer
3. Note the **IP Address** displayed in the app (e.g., `192.168.1.100`)
4. Note the **Port** number (default is `4747`)

### 3. Configure cameras.json

Edit `config/cameras.json` with your DroidCam details:

```json
[
  {
    "id": "droid_cam_1",
    "name": "Living Room Camera",
    "ip_address": "192.168.1.100",
    "port": 4747
  },
  {
    "id": "droid_cam_2",
    "name": "Entry Point Camera",
    "ip_address": "192.168.1.101",
    "port": 4747
  }
]
```

**Replace the IP addresses with your actual smartphone IP addresses!**

### 4. Test Connection

Before starting the application, test if your DroidCam is accessible:

```python
from src.droidcam_manager import DroidCamManager

manager = DroidCamManager()
if manager.test_connection("192.168.1.100", 4747):
    print("✅ DroidCam connection successful!")
else:
    print("❌ Failed to connect to DroidCam")
```

### 5. Run the Application

```bash
python app_flask.py
```

## Tips for Best Performance

### Optimize FPS (Frames Per Second)

1. **Network Optimization:**
   - Use a 5GHz WiFi network for better bandwidth
   - Minimize distance between phone and router
   - Reduce interference from other devices

2. **DroidCam App Settings:**
   - Set resolution to 720p (balances quality and FPS)
   - Disable auto-focus for faster frames
   - Use higher FPS setting in app (if available)

3. **Application Settings:**
   - Reduce frame size for processing (check `src/utils.py`)
   - Enable frame skipping for real-time detection
   - Adjust the buffer size in `droidcam_manager.py`:
     ```python
     # Line 59 in droidcam_manager.py
     self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Lower = faster but may drop frames
     ```

### Reduce Latency

1. Keep DroidCam app open and active on phone
2. Use WiFi 5GHz instead of 2.4GHz
3. Disable other video streaming apps
4. Position phone strategically for consistent signal

## Troubleshooting

### DroidCam Connection Failed

**Problem:** "Failed to connect to DroidCam (URL)"

**Solutions:**
1. Verify phone is connected to WiFi
2. Check IP address is correct (`192.168.x.x` format)
3. Ensure port is `4747` (default)
4. Disable phone's firewall temporarily
5. On computer: Check firewall allows outbound connections
6. Ping the phone from command line:
   ```bash
   ping 192.168.1.100
   ```

### Low FPS / Slow Detection

**Problem:** Video is lagging or detection is slow

**Solutions:**
1. Check WiFi signal strength on phone
2. Move phone closer to router
3. Reduce resolution in DroidCam app settings
4. Close other apps on phone
5. Check network bandwidth usage
6. Monitor CPU usage on computer

### Video Freezes or Disconnects

**Problem:** Video stream keeps dropping

**Solutions:**
1. Keep DroidCam app in foreground (don't minimize)
2. Disable phone's auto-lock/sleep mode
3. Check WiFi stability (move closer to router)
4. Restart DroidCam app
5. Check system logs for error messages

## Network Setup Tips

### Find Your Phone's IP Address

In DroidCam app:
1. Open the app
2. Look for "IP" or "Connection" section
3. Your IP will be displayed (e.g., `192.168.1.100`)

Alternatively, from command line:
```bash
# Windows PowerShell
Get-NetNeighbor | Where-Object {$_.State -eq "Reachable"} | Select IPAddress
```

### Multiple Cameras on Same Network

You can connect multiple phones running DroidCam:
1. Each phone gets a unique IP address (last digit differs)
2. Add each to `cameras.json` with different IP addresses
3. Application will automatically balance detection across all cameras

## Advanced Configuration

### Increase Resolution for Better Detection

Edit `droidcam_manager.py`:

```python
# In DroidCam.connect() method
self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
self.capture.set(cv2.CAP_PROP_FPS, 30)
```

### Custom Port (If Needed)

DroidCam can use custom ports if configured in the app:

```json
{
  "id": "droid_cam_custom",
  "name": "Custom Port Camera",
  "ip_address": "192.168.1.100",
  "port": 8080
}
```

## Performance Comparison

| Aspect | IP Cameras | DroidCam |
|--------|-----------|----------|
| Setup Time | Complex | 2-3 minutes |
| Cost | High | Free/Low |
| FPS | 20-30 | 25-30+ |
| Latency | 500-1000ms | 100-300ms |
| Flexibility | Fixed location | Mobile |
| Network | Dedicated | WiFi |

## Next Steps

1. Install DroidCam on your phones
2. Note down IP addresses
3. Update `config/cameras.json`
4. Run the application
5. Monitor detection logs in `data/detection_log.csv`

For more DroidCam info: https://www.dev47apps.com/
