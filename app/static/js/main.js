// My OCR App Frontend

// DOM Elements
const apiKeyInput = document.getElementById('api-key');
const saveKeyBtn = document.getElementById('save-key');
const authStatus = document.getElementById('auth-status');
const cameraStatus = document.getElementById('camera-status');
const cameraBtn = document.getElementById('camera-btn');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const cameraVideo = document.getElementById('camera-video');
const previewContainer = document.getElementById('preview-container');
const cameraControls = document.getElementById('camera-controls');
const captureBtn = document.getElementById('capture-btn');
const stopCameraBtn = document.getElementById('stop-camera-btn');
const modelSelect = document.getElementById('model-select');
const minConfidenceSlider = document.getElementById('min-confidence');
const confidenceValue = document.getElementById('confidence-value');
const processBtn = document.getElementById('process-btn');
const processStatus = document.getElementById('process-status');
const resultsContainer = document.getElementById('results-container');
const extractedText = document.getElementById('extracted-text');
const processingDetails = document.getElementById('processing-details');

// State
let mediaStream = null;
let currentImage = null;
let apiKey = sessionStorage.getItem('ocr-api-key') || '';

// Initialize
function init() {
    // Load saved API key
    if (apiKey) {
        apiKeyInput.value = apiKey;
        updateAuthStatus('API key loaded', 'success');
    }
    
    // Set up event listeners
    saveKeyBtn.addEventListener('click', saveApiKey);
    cameraBtn.addEventListener('click', startCamera);
    fileInput.addEventListener('change', handleFileUpload);
    captureBtn.addEventListener('click', captureFromCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    minConfidenceSlider.addEventListener('input', updateConfidenceValue);
    processBtn.addEventListener('click', processImage);
    
    // Update confidence value on load
    updateConfidenceValue();
}

// Save API key for the current browser session only
function saveApiKey() {
    const key = apiKeyInput.value.trim();
    if (!key) {
        updateAuthStatus('API key cannot be empty', 'error');
        return;
    }

    apiKey = key;
    sessionStorage.setItem('ocr-api-key', key);
    updateAuthStatus('API key saved for this session', 'success');
}

// Update a status element with optional auto-hide for success messages
function updateStatus(element, message, type) {
    element.textContent = message;
    element.className = `status ${type}`;

    if (type === 'success') {
        setTimeout(() => {
            if (element.textContent === message) {
                element.textContent = '';
                element.className = 'status';
            }
        }, 3000);
    }
}

function updateAuthStatus(message, type) {
    updateStatus(authStatus, message, type);
}

function updateCameraStatus(message, type) {
    updateStatus(cameraStatus, message, type);
}

// Start camera
async function startCamera() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraVideo.srcObject = mediaStream;
        cameraVideo.classList.add('show');
        previewImg.classList.remove('show');
        cameraControls.style.display = 'block';
        
        // Hide other preview elements
        previewContainer.style.height = '300px';
        
        updateCameraStatus('Camera started', 'success');
    } catch (err) {
        console.error('Error accessing camera:', err);
        updateCameraStatus(`Camera error: ${err.message}`, 'error');
    }
}

// Stop camera
function stopCamera() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
        cameraVideo.srcObject = null;
        cameraVideo.classList.remove('show');
        cameraControls.style.display = 'none';
        
        // Show placeholder
        previewContainer.style.height = 'auto';
        updateCameraStatus('Camera stopped', 'success');
    }
}

// Capture image from camera
function captureFromCamera() {
    if (!cameraVideo.srcObject) return;
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    
    context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob
    canvas.toBlob(blob => {
        currentImage = blob;
        
        // Display preview
        const url = URL.createObjectURL(blob);
        previewImg.src = url;
        previewImg.classList.add('show');
        cameraVideo.classList.remove('show');
        
        updateCameraStatus('Image captured from camera', 'success');
    }, 'image/jpeg', 0.9);
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Check if it's an image
    if (!file.type.match('image.*')) {
        updateCameraStatus('Please select an image file', 'error');
        return;
    }
    
    currentImage = file;
    
    // Display preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        previewImg.classList.add('show');
        cameraVideo.classList.remove('show');
        cameraControls.style.display = 'none';
        
        updateCameraStatus(`File loaded: ${file.name}`, 'success');
    };
    reader.readAsDataURL(file);
}

// Update confidence value display
function updateConfidenceValue() {
    const value = parseFloat(minConfidenceSlider.value);
    confidenceValue.textContent = value.toFixed(1);
}

// Process image with OCR
async function processImage() {
    if (!currentImage) {
        updateProcessStatus('Please select or capture an image first', 'error');
        return;
    }
    
    if (!apiKey) {
        updateProcessStatus('Please enter and save your API key first', 'error');
        return;
    }
    
    updateProcessStatus('Processing...', 'warning');
    processBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', currentImage);
        
        const model = modelSelect.value;
        const minConfidence = parseFloat(minConfidenceSlider.value);
        
        let url = '/ocr';
        const params = new URLSearchParams();
        if (model && model !== 'paddle') {
            params.append('model', model);
        }
        if (minConfidence > 0) {
            params.append('min_confidence', minConfidence.toString());
        }
        if (params.toString()) {
            url += '?' + params.toString();
        }
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`
            },
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
        updateProcessStatus('Processing completed successfully!', 'success');
        
    } catch (error) {
        console.error('OCR processing error:', error);
        updateProcessStatus(`Error: ${error.message}`, 'error');
    } finally {
        processBtn.disabled = false;
    }
}

// Update process status display
function updateProcessStatus(message, type) {
    processStatus.textContent = message;
    processStatus.className = `status ${type}`;
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            if (processStatus.textContent === message) {
                processStatus.textContent = '';
                processStatus.className = 'status';
            }
        }, 5000);
    }
}

// Display OCR results
function displayResults(result) {
    extractedText.textContent = result.text || 'No text found';
    processingDetails.textContent = '';

    const rows = [
        ['Model', result.model?.label || result.model?.id || 'Unknown'],
        ['Lines', String(result.line_count || 0)],
        ['Processing Time', `${result.processing_time_ms || 0} ms`],
        ['Image Size', `${result.image?.width || 0} × ${result.image?.height || 0} pixels`],
    ];

    if (result.usage) {
        rows.push([
            'Token Usage',
            `${result.usage.input_tokens || 0} input, ${result.usage.output_tokens || 0} output`,
        ]);
        const cost = result.usage.estimated_cost_usd;
        rows.push(['Estimated Cost', cost === null ? 'unavailable' : `$${cost.toFixed(6)}`]);
    }

    for (const [label, value] of rows) {
        const p = document.createElement('p');
        const strong = document.createElement('strong');
        strong.textContent = `${label}: `;
        p.appendChild(strong);
        p.appendChild(document.createTextNode(value));
        processingDetails.appendChild(p);
    }

    resultsContainer.style.display = 'block';
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);