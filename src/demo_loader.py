"""
Demo File Loader
Handles downloading and caching demo videos from YouTube and direct demo images.
"""

import os
import subprocess
import requests
from PIL import Image
import io


class DemoLoader:
    """Load and cache demo videos and images"""
    
    def __init__(self, cache_dir="data/demo_cache"):
        """
        Initialize demo loader
        
        Args:
            cache_dir (str): Directory to cache downloaded files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Demo sources
        self.demo_videos = {
            "Sample Video 1": "https://www.youtube.com/watch?v=9c-DrMe8o5Q",
            "Sample Video 2": "https://www.youtube.com/watch?v=Rbp2XUSeUNE",
        }
        
        # Direct image URLs provided by user
        self.demo_images = {
            "Demo Image 1": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQc26-dZZVFPt06CGm5VJKpkkDANuNC9etTZenjterekoAH0XUZGiw9uTgdffhW3ctlkUjF2EPl76w9ltiF_Co14XEY0OoNCr8-axYt-A&s=10",
            "Demo Image 2": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNZpw5rj_AI5WWxV40YiP7kzI30DEgMRLGkSsOIYRn3vCA4zimdKmVP39mwAV5OYhY3h0thsqYrKBgUQsBogMEeJOSMI-FFks2zB8L3cg&s=10",
        }
    
    def get_demo_videos(self):
        """
        Get list of available demo videos
        
        Returns:
            dict: {name: file_path}
        """
        available_videos = {}
        
        for name, url in self.demo_videos.items():
            video_path = self._get_cached_video(name, url)
            if video_path:
                available_videos[name] = video_path
        
        return available_videos
    
    def get_demo_images(self):
        """
        Get list of available demo images
        
        Returns:
            dict: {name: image_path_or_pil_image}
        """
        available_images = {}
        
        for name, url in self.demo_images.items():
            try:
                image_path = self._get_cached_image(name, url)
                if image_path:
                    available_images[name] = image_path
                else:
                    # Fallback: keep direct URL available to UI even if cache fails.
                    available_images[name] = url
            except Exception as e:
                print(f"⚠️ Could not load demo image {name}: {e}")
                available_images[name] = url
        
        return available_images
    
    def _get_cached_video(self, name, youtube_url):
        """
        Download and cache YouTube video
        
        Args:
            name (str): Demo name
            youtube_url (str): YouTube URL
            
        Returns:
            str: Path to cached video file
        """
        # Cache file path
        video_filename = f"{name.replace(' ', '_').lower()}.mp4"
        cache_path = os.path.join(self.cache_dir, video_filename)
        
        # Return if already cached
        if os.path.exists(cache_path):
            print(f"✅ Using cached video: {name}")
            return cache_path
        
        try:
            print(f"📥 Downloading demo video: {name}...")
            
            # Preferred: use Python yt_dlp API (works even if yt-dlp binary is missing)
            try:
                import yt_dlp

                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'outtmpl': cache_path,
                    'noplaylist': True,
                    'retries': 3,
                    'socket_timeout': 30,
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])

                if os.path.exists(cache_path):
                    print(f"✅ Downloaded demo video: {name}")
                    return cache_path
            except Exception as api_err:
                print(f"⚠️ yt_dlp API failed for {name}: {api_err}")

            # Fallback: use yt-dlp CLI if available
            cmd = [
                'yt-dlp',
                '--no-playlist',
                '--socket-timeout', '30',
                '--retries', '3',
                '-f', 'best[ext=mp4]/best',
                '-o', cache_path,
                youtube_url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)

            if result.returncode == 0 and os.path.exists(cache_path):
                print(f"✅ Downloaded demo video: {name}")
                return cache_path

            print(f"❌ Failed to download {name}: {result.stderr}")
            return None
                
        except FileNotFoundError:
            print("⚠️ yt-dlp not found. Install with: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            print(f"❌ Download timeout for {name}")
            return None
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            return None
    
    def _get_cached_image(self, name, image_url):
        """
        Download and cache image from direct URL
        
        Args:
            name (str): Demo name
            image_url (str): Direct image URL
            
        Returns:
            str: Path to cached image file
        """
        image_filename = f"{name.replace(' ', '_').lower()}.jpg"
        cache_path = os.path.join(self.cache_dir, image_filename)
        
        # Return if already cached
        if os.path.exists(cache_path):
            print(f"✅ Using cached image: {name}")
            return cache_path
        
        try:
            print(f"📥 Fetching demo image: {name}...")
            image_data = self._download_image(image_url)
            
            if image_data:
                # Save to cache
                with open(cache_path, 'wb') as f:
                    f.write(image_data)
                print(f"✅ Cached demo image: {name}")
                return cache_path
            else:
                print(f"❌ Could not extract image from {name}")
                return None
        
        except Exception as e:
            print(f"❌ Error caching image {name}: {e}")
            return None
    
    def _download_image(self, image_url, max_size=5*1024*1024):
        """
        Download image from URL
        
        Args:
            image_url (str): Image URL
            max_size (int): Maximum file size in bytes
            
        Returns:
            bytes: Image data or None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size:
                print(f"Image too large: {content_length} bytes")
                return None
            
            image_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk
                if len(image_data) > max_size:
                    print(f"Image download exceeded max size")
                    return None
            
            # Verify it's a valid image
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            
            # Re-open to get data (verify closes the file object)
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > 1920 or img.size[1] > 1080:
                img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            
            # Save and return
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85)
            return output.getvalue()
        
        except Exception as e:
            print(f"Error downloading image from {image_url}: {e}")
            return None
    
    def get_all_demos(self):
        """
        Get all available demos
        
        Returns:
            dict: {
                'videos': {name: path},
                'images': {name: path}
            }
        """
        return {
            'videos': self.get_demo_videos(),
            'images': self.get_demo_images()
        }
