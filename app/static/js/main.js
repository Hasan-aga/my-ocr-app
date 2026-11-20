// My OCR App Frontend

// DOM Elements
const apiKeyInput = document.getElementById('api-key');
const saveKeyBtn = document.getElementById('save-key');
const authStatus = document.getElementById('auth-status');
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
let apiKey = localStorage.getItem('ocr-api-key') || '';

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

// Save API key to localStorage
function saveApiKey() {
    const key = apiKeyInput.value.trim();
    if (!key) {
        updateAuthStatus('API key cannot be empty', 'error');
        return;
    }
    
    apiKey = key;
    localStorage.setItem('ocr-api-key', key);
    updateAuthStatus('API key saved successfully', 'success');
}

// Update authentication status display
function updateAuthStatus(message, type) {
    authStatus.textContent = message;
    authStatus.className = `status ${type}`;
    
    // Auto-hide success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            if (authStatus.textContent === message) {
                authStatus.textContent = '';
                authStatus.className = 'status';
            }
        }, 3000);
    }
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
        
        updateAuthStatus('Camera started', 'success');
    } catch (err) {
        console.error('Error accessing camera:', err);
        updateAuthStatus(`Camera error: ${err.message}`, 'error');
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
        updateAuthStatus('Camera stopped', 'success');
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
        
        updateAuthStatus('Image captured from camera', 'success');
    }, 'image/jpeg', 0.9);
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Check if it's an image
    if (!file.type.match('image.*')) {
        updateAuthStatus('Please select an image file', 'error');
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
        
        updateAuthStatus(`File loaded: ${file.name}`, 'success');
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
    // Clear previous results
    extractedText.textContent = '';
    processingDetails.innerHTML = '';
    
    // Show extracted text
    extractedText.textContent = result.text || 'No text found';
    
    // Show processing details
    const detailsHTML = `
        <p><strong>Model:</strong> ${result.model?.label || result.model?.id || 'Unknown'}</p>
        <p><strong>Lines:</strong> ${result.line_count || 0}</p>
        <p><strong>Processing Time:</strong> ${result.processing_time_ms || 0} ms</p>
        <p><strong>Image Size:</strong> ${result.image?.width || 0} × ${result.image?.height || 0} pixels</p>
    `;
    
    if (result.usage) {
        detailsHTML += `
            <p><strong>Token Usage:</strong> ${result.usage.input_tokens || 0} input, ${result.usage.output_tokens || 0} output</p>
            <p><strong>Estimated Cost:</strong> $${(result.usage.estimated_cost_usd || 0).toFixed(6)}</p>
        `;
    }
    
    processingDetails.innerHTML = detailsHTML;
    
    // Show results container
    resultsContainer.style.display = 'block';
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);