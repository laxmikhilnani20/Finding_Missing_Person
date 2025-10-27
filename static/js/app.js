// Initialize Socket.IO
const socket = io();

// State
let isMonitoring = false;
let cameras = {};
let persons = {};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadCameras();
    loadPersons();
    loadDetectionLog();
    setupSocketListeners();
    setupThresholdSlider();
    setupImagePreview();
});

// Socket.IO Listeners
function setupSocketListeners() {
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('frame_update', (data) => {
        updateCameraFeed(data.camera_id, data.frame, data.has_detection);
    });

    socket.on('detection_alert', (data) => {
        showAlert(`🚨 ${data.person_name} detected at ${data.camera_name}! Confidence: ${(data.similarity * 100).toFixed(1)}%`);
        playAlertSound();
        updateDetectionCount();
        loadDetectionLog();
    });
}

// API Calls
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {}
    };

    if (body && !(body instanceof FormData)) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
    } else if (body) {
        options.body = body;
    }

    try {
        const response = await fetch(endpoint, options);
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showAlert('Error communicating with server', 'danger');
        return { success: false, message: error.message };
    }
}

// Camera Management
async function loadCameras() {
    const result = await apiCall('/api/cameras');
    if (result.success) {
        cameras = result.cameras;
        renderCameraList();
        updateStats();
    }
}

function renderCameraList() {
    const container = document.getElementById('camera-list');
    container.innerHTML = '';

    if (Object.keys(cameras).length === 0) {
        container.innerHTML = '<p class="text-secondary">No cameras added</p>';
        return;
    }

    Object.entries(cameras).forEach(([id, info]) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML = `
            <div class="list-item-content">
                <div class="list-item-title">${info.name}</div>
                <div class="list-item-subtitle">
                    ${info.is_active ? '🟢 Active' : '🔴 Inactive'} | FPS: ${info.fps}
                </div>
            </div>
            <div class="list-item-actions">
                <button class="icon-btn danger" onclick="removeCamera('${id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        container.appendChild(item);
    });
}

async function addCamera() {
    const name = document.getElementById('camera-name').value.trim();
    const url = document.getElementById('camera-url').value.trim();

    if (!name || !url) {
        showAlert('Please enter camera name and URL', 'warning');
        return;
    }

    const result = await apiCall('/api/cameras/add', 'POST', { name, url });
    
    if (result.success) {
        showAlert(result.message, 'success');
        closeModal('camera-modal');
        document.getElementById('camera-name').value = '';
        document.getElementById('camera-url').value = '';
        loadCameras();
    } else {
        showAlert(result.message, 'danger');
    }
}

async function testCamera() {
    const url = document.getElementById('camera-url').value.trim();

    if (!url) {
        showAlert('Please enter camera URL', 'warning');
        return;
    }

    showAlert('Testing connection...', 'info');
    const result = await apiCall('/api/cameras/test', 'POST', { url });
    
    if (result.success) {
        showAlert('✅ Connection successful!', 'success');
    } else {
        showAlert('❌ Connection failed', 'danger');
    }
}

async function removeCamera(cameraId) {
    if (!confirm('Are you sure you want to remove this camera?')) {
        return;
    }

    const result = await apiCall(`/api/cameras/remove/${cameraId}`, 'DELETE');
    
    if (result.success) {
        showAlert(result.message, 'success');
        loadCameras();
    } else {
        showAlert(result.message, 'danger');
    }
}

// Person Management
async function loadPersons() {
    const result = await apiCall('/api/persons');
    if (result.success) {
        persons = result.persons;
        renderPersonList();
        updateStats();
    }
}

function renderPersonList() {
    const container = document.getElementById('person-list');
    container.innerHTML = '';

    if (persons.length === 0) {
        container.innerHTML = '<p class="text-secondary">No persons registered</p>';
        return;
    }

    persons.forEach(person => {
        const item = document.createElement('div');
        item.className = 'list-item person-item';
        item.innerHTML = `
            ${person.image ? `<img src="data:image/jpeg;base64,${person.image}" class="person-image" alt="${person.name}">` : '<div class="person-image"></div>'}
            <div class="list-item-content">
                <div class="list-item-title">${person.name}</div>
            </div>
            <div class="list-item-actions">
                <button class="icon-btn danger" onclick="removePerson('${person.name}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        container.appendChild(item);
    });
}

async function addPerson() {
    const name = document.getElementById('person-name').value.trim();
    const imageInput = document.getElementById('person-image');

    if (!name || !imageInput.files[0]) {
        showAlert('Please enter name and upload photo', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('name', name);
    formData.append('image', imageInput.files[0]);

    const result = await apiCall('/api/persons/add', 'POST', formData);
    
    if (result.success) {
        showAlert(result.message, 'success');
        closeModal('person-modal');
        document.getElementById('person-name').value = '';
        document.getElementById('person-image').value = '';
        document.getElementById('image-preview').innerHTML = '';
        loadPersons();
    } else {
        showAlert(result.message, 'danger');
    }
}

async function removePerson(personName) {
    if (!confirm(`Are you sure you want to remove ${personName}?`)) {
        return;
    }

    const result = await apiCall(`/api/persons/remove/${personName}`, 'DELETE');
    
    if (result.success) {
        showAlert(result.message, 'success');
        loadPersons();
    } else {
        showAlert(result.message, 'danger');
    }
}

// Monitoring Control
async function startMonitoring() {
    const result = await apiCall('/api/monitoring/start', 'POST');
    
    if (result.success) {
        isMonitoring = true;
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        document.getElementById('no-monitoring').style.display = 'none';
        showAlert(result.message, 'success');
        initializeCameraFeeds();
    } else {
        showAlert(result.message, 'danger');
    }
}

async function stopMonitoring() {
    const result = await apiCall('/api/monitoring/stop', 'POST');
    
    if (result.success) {
        isMonitoring = false;
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('no-monitoring').style.display = 'block';
        document.getElementById('camera-feeds').innerHTML = '';
        showAlert(result.message, 'info');
    } else {
        showAlert(result.message, 'danger');
    }
}

function initializeCameraFeeds() {
    const container = document.getElementById('camera-feeds');
    container.innerHTML = '';

    Object.entries(cameras).forEach(([id, info]) => {
        const feed = document.createElement('div');
        feed.className = 'camera-feed';
        feed.id = `feed-${id}`;
        feed.innerHTML = `
            <div class="camera-feed-header">
                <span>${info.name}</span>
                <div class="camera-status">
                    <span class="status-dot"></span>
                    <span>Searching...</span>
                </div>
            </div>
            <div class="camera-feed-body">
                <img id="img-${id}" class="camera-image" src="" alt="${info.name}">
            </div>
        `;
        container.appendChild(feed);
    });
}

function updateCameraFeed(cameraId, frameBase64, hasDetection) {
    const img = document.getElementById(`img-${cameraId}`);
    const feed = document.getElementById(`feed-${cameraId}`);
    
    if (img && feed) {
        img.src = `data:image/jpeg;base64,${frameBase64}`;
        
        if (hasDetection) {
            feed.classList.add('detected');
        } else {
            feed.classList.remove('detected');
        }
    }
}

// Threshold Management
function setupThresholdSlider() {
    const slider = document.getElementById('threshold-slider');
    const valueDisplay = document.getElementById('threshold-value');

    slider.addEventListener('input', async (e) => {
        const value = parseFloat(e.target.value);
        valueDisplay.textContent = value.toFixed(2);
        await apiCall('/api/threshold/update', 'POST', { threshold: value });
    });
}

// Detection Log
async function loadDetectionLog() {
    const result = await apiCall('/api/detections/log?limit=50');
    
    if (result.success) {
        renderDetectionTable(result.detections);
        renderDetectionStats(result);
        renderRecentImages(result.detections.slice(-6));
    }
}

function renderDetectionTable(detections) {
    const tbody = document.getElementById('detection-table-body');
    tbody.innerHTML = '';

    if (detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center">No detections logged yet</td></tr>';
        return;
    }

    detections.reverse().forEach(detection => {
        const confidence = detection.similarity * 100;
        let badgeClass = 'confidence-low';
        if (confidence >= 80) badgeClass = 'confidence-high';
        else if (confidence >= 65) badgeClass = 'confidence-medium';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${detection.timestamp}</td>
            <td>${detection.person_name}</td>
            <td>${detection.camera_name}</td>
            <td><span class="confidence-badge ${badgeClass}">${confidence.toFixed(1)}%</span></td>
            <td>
                <button class="icon-btn" onclick="viewImage('${detection.frame_path}')">
                    <i class="fas fa-image"></i>
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function renderDetectionStats(data) {
    document.getElementById('total-detections').textContent = data.total;
    document.getElementById('unique-persons-detected').textContent = data.unique_persons;
    document.getElementById('avg-confidence').textContent = `${(data.avg_confidence * 100).toFixed(1)}%`;
}

function renderRecentImages(detections) {
    const container = document.getElementById('recent-images-grid');
    container.innerHTML = '';

    if (detections.length === 0) {
        container.innerHTML = '<p class="text-secondary">No recent detections</p>';
        return;
    }

    detections.forEach(detection => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.onclick = () => viewImage(detection.frame_path);
        card.innerHTML = `
            <img src="/api/detections/image/${detection.frame_path}" alt="${detection.person_name}">
            <div class="image-card-info">
                <div class="image-card-title">${detection.person_name}</div>
                <div class="image-card-subtitle">${detection.timestamp}</div>
            </div>
        `;
        container.appendChild(card);
    });
}

async function exportReport() {
    try {
        window.location.href = '/api/detections/export';
        showAlert('Report exported successfully', 'success');
    } catch (error) {
        showAlert('Failed to export report', 'danger');
    }
}

// UI Functions
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');

    // Load detection log if detections tab is shown
    if (tabName === 'detections') {
        loadDetectionLog();
    }
}

function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function viewImage(imagePath) {
    const img = document.getElementById('modal-image');
    img.src = `/api/detections/image/${imagePath}`;
    openModal('image-modal');
}

function showAlert(message, type = 'info') {
    const banner = document.getElementById('alert-banner');
    const messageEl = document.getElementById('alert-message');
    
    // Set colors based on type
    const colors = {
        success: '#22c55e',
        danger: '#ef4444',
        warning: '#f59e0b',
        info: '#06b6d4'
    };
    
    banner.style.background = colors[type] || colors.info;
    messageEl.textContent = message;
    banner.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        banner.classList.add('hidden');
    }, 5000);
}

function closeAlert() {
    document.getElementById('alert-banner').classList.add('hidden');
}

function updateStats() {
    document.getElementById('active-cameras').textContent = Object.keys(cameras).length;
    document.getElementById('registered-persons').textContent = persons.length;
}

async function updateDetectionCount() {
    const result = await apiCall('/api/monitoring/status');
    if (result.success) {
        document.getElementById('detection-count').textContent = result.detection_count;
    }
}

function setupImagePreview() {
    const imageInput = document.getElementById('person-image');
    const preview = document.getElementById('image-preview');

    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }
    });
}

function playAlertSound() {
    // Create audio context for alert sound
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    } catch (error) {
        console.log('Audio not supported');
    }
}

// Close modals on outside click
window.onclick = (event) => {
    if (event.target.classList.contains('modal')) {
        event.target.classList.remove('active');
    }
}

// Periodic updates
setInterval(() => {
    if (!isMonitoring) {
        updateDetectionCount();
    }
}, 5000);