// Global State
let selectedFile = null;
let audioContext = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingStream = null;
let recordTimerInterval = null;
let recordStartTime = null;
let animationFrameId = null;
let audioBufferSource = null;
let isRecording = false;

// Audio Recording Buffers (for custom WAV encoder)
let leftchannel = [];
let recordingLength = 0;
let recSampleRate = 16000;

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const selectedFileCard = document.getElementById('selected-file-card');
const fileNameEl = document.getElementById('file-name');
const fileSizeEl = document.getElementById('file-size');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsPanel = document.getElementById('results-panel');
const resultsEmpty = document.getElementById('results-empty');
const loadingState = document.getElementById('loading-state');
const resultsContent = document.getElementById('results-content');
const audioPreview = document.getElementById('audio-preview');

// Tabs Switching
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    if (tab === 'upload') {
        document.getElementById('tab-upload').classList.add('active');
        document.getElementById('content-upload').classList.add('active');
        stopRecordingState();
    } else {
        document.getElementById('tab-record').classList.add('active');
        document.getElementById('content-record').classList.add('active');
    }
}

// --- FILE UPLOAD LOGIC ---
if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file (WAV, FLAC, MP3, etc.).');
        return;
    }
    
    selectedFile = file;
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    
    selectedFileCard.style.display = 'flex';
    analyzeBtn.disabled = false;
    
    // Set audio preview src
    const fileURL = URL.createObjectURL(file);
    audioPreview.src = fileURL;
}

function resetFile() {
    selectedFile = null;
    if (fileInput) fileInput.value = '';
    selectedFileCard.style.display = 'none';
    analyzeBtn.disabled = true;
    audioPreview.src = '';
}

// --- MICROPHONE RECORDING LOGIC ---
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    isRecording = true;
    audioChunks = [];
    leftchannel = [];
    recordingLength = 0;
    
    const recordBtn = document.getElementById('record-btn');
    const recordIcon = document.getElementById('record-icon');
    const recordStatus = document.getElementById('record-status');
    const timerEl = document.getElementById('record-timer');
    
    recordBtn.classList.add('recording');
    recordIcon.className = 'fa-solid fa-square icon-record';
    recordStatus.textContent = 'Recording live...';
    
    try {
        recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        
        // Setup Web Audio API for visualization and resampling
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: recSampleRate // Request 16000Hz sampling rate from browser
        });
        
        const source = audioContext.createMediaStreamSource(recordingStream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        
        // Setup ScriptProcessor for saving samples to WAV
        // 2048 frame size, 1 input channel, 1 output channel
        const bufferSize = 2048;
        const recorderNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        recorderNode.onaudioprocess = function(e) {
            if (!isRecording) return;
            const left = e.inputBuffer.getChannelData(0);
            // Clone the buffer
            leftchannel.push(new Float32Array(left));
            recordingLength += bufferSize;
        };
        
        source.connect(recorderNode);
        recorderNode.connect(audioContext.destination);
        
        // Start Canvas visualizer
        visualize(analyser);
        
        // Start Timer
        recordStartTime = Date.now();
        timerEl.textContent = '00:00';
        recordTimerInterval = setInterval(() => {
            const elapsed = Date.now() - recordStartTime;
            const seconds = Math.floor((elapsed / 1000) % 60);
            const minutes = Math.floor((elapsed / 1000 / 60) % 60);
            timerEl.textContent = 
                (minutes < 10 ? '0' + minutes : minutes) + ':' + 
                (seconds < 10 ? '0' + seconds : seconds);
        }, 1000);
        
    } catch (err) {
        console.error('Microphone access denied or error:', err);
        alert('Could not access microphone: ' + err.message);
        stopRecordingState();
    }
}

function stopRecording() {
    isRecording = false;
    
    // Stop recording tracks
    if (recordingStream) {
        recordingStream.getTracks().forEach(track => track.stop());
    }
    
    // Stop timer
    if (recordTimerInterval) {
        clearInterval(recordTimerInterval);
    }
    
    // Stop visualizer animation
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    // Save recording buffers into WAV
    if (recordingLength > 0) {
        // Flatten the recording buffer
        const flatBuffer = flattenBuffer(leftchannel, recordingLength);
        // Encode to WAV Blob
        const wavBlob = bufferToWav(flatBuffer, recSampleRate);
        
        // Create selected file
        const file = new File([wavBlob], `recorded_audio_${Date.now()}.wav`, { type: 'audio/wav' });
        handleFile(file);
    }
    
    stopRecordingState();
}

function stopRecordingState() {
    isRecording = false;
    const recordBtn = document.getElementById('record-btn');
    const recordIcon = document.getElementById('record-icon');
    const recordStatus = document.getElementById('record-status');
    
    if (recordBtn) recordBtn.classList.remove('recording');
    if (recordIcon) recordIcon.className = 'fa-solid fa-circle icon-record';
    if (recordStatus) recordStatus.textContent = 'Recording saved. Ready for analysis.';
    
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
    }
}

// Flattening array of float buffers
function flattenBuffer(channelBuffer, recordingLength) {
    const result = new Float32Array(recordingLength);
    let offset = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        const buffer = channelBuffer[i];
        result.set(buffer, offset);
        offset += buffer.length;
    }
    return result;
}

// WAV encoding helper functions
function bufferToWav(buffer, sampleRate) {
    const bufferLength = buffer.length;
    const wavBuffer = new ArrayBuffer(44 + bufferLength * 2);
    const view = new DataView(wavBuffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + bufferLength * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, bufferLength * 2, true);

    floatTo16BitPCM(view, 44, buffer);

    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// --- AUDIO REAL-TIME CANVAS VISUALIZATION ---
function visualize(analyser) {
    const canvas = document.getElementById('visualizer');
    if (!canvas) return;
    const canvasCtx = canvas.getContext('2d');
    
    // Match visualizer canvas layout bounds
    const resizeCanvas = () => {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 80;
    };
    resizeCanvas();
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    
    function draw() {
        if (!isRecording) return;
        
        animationFrameId = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        
        canvasCtx.fillStyle = 'rgba(10, 14, 26, 0.4)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        canvasCtx.lineWidth = 2;
        // Draw elegant violet/blue line
        const grad = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
        grad.addColorStop(0, '#6366f1');
        grad.addColorStop(1, '#8b5cf6');
        canvasCtx.strokeStyle = grad;
        
        canvasCtx.beginPath();
        
        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            
            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }
    
    draw();
}

// --- MODEL INFERENCE PREDICT INTEGRATION ---
async function analyzeAudio() {
    if (!selectedFile) return;
    
    // Show Loading
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'none';
    loadingState.style.display = 'flex';
    
    // Clear and prepare request body
    const formData = new FormData();
    formData.append('audio', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server error occurred');
        }
        
        const result = await response.json();
        
        if (result.success) {
            renderResults(result);
        } else {
            throw new Error('Analysis request failed');
        }
        
    } catch (error) {
        console.error('Error during analysis:', error);
        alert('Analysis Error: ' + error.message);
        
        // Reset view
        loadingState.style.display = 'none';
        resultsEmpty.style.display = 'flex';
    }
}

function renderResults(result) {
    loadingState.style.display = 'none';
    resultsContent.style.display = 'flex';
    
    const riskBadge = document.getElementById('risk-badge');
    const verdictCard = document.getElementById('verdict-card');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictValue = document.getElementById('verdict-value');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const spoofScoreValue = document.getElementById('spoof-score-value');
    const scoreDot = document.getElementById('score-dot');
    const securityStatusText = document.getElementById('security-status-text');
    
    // 1. Verdict display styling
    if (result.verdict === 'spoof') {
        riskBadge.textContent = `${result.risk_level} Risk`;
        riskBadge.className = 'badge threat';
        
        verdictCard.className = 'verdict-card threat';
        verdictIcon.className = 'fa-solid fa-triangle-exclamation verdict-icon';
        verdictValue.textContent = 'SPOOFED / DEEPFAKE';
        
        confidenceBar.className = 'progress-bar threat';
        
        securityStatusText.textContent = `Vulnerability detected. The sample shows standard acoustic anomalies indicating synthetic generation (Score ${result.spoof_prob.toFixed(4)} is above the threat threshold).`;
    } else {
        riskBadge.textContent = 'Safe';
        riskBadge.className = 'badge safe';
        
        verdictCard.className = 'verdict-card safe';
        verdictIcon.className = 'fa-solid fa-circle-check verdict-icon';
        verdictValue.textContent = 'BONAFIDE / NATURAL';
        
        confidenceBar.className = 'progress-bar safe';
        
        securityStatusText.textContent = `Acoustic profiles verified. The sample contains normal human vocal structures with no signature artificial voice fingerprints.`;
    }
    
    // 2. Confidence percentage & progress bar
    confidenceValue.textContent = `${result.confidence.toFixed(1)}%`;
    // Delay width change for smooth transition animation
    setTimeout(() => {
        confidenceBar.style.width = `${result.confidence}%`;
    }, 100);
    
    // 3. Raw spoof score
    spoofScoreValue.textContent = result.spoof_prob.toFixed(4);
    
    // 4. Dot position mapping (split scale around threshold at 50%)
    // 0 to threshold (0.0651) -> 0% to 50%
    // threshold to 1.0 -> 50% to 100%
    let dotPos = 0;
    const thresh = result.threshold;
    if (result.spoof_prob > thresh) {
        dotPos = 50 + ((result.spoof_prob - thresh) / (1 - thresh)) * 50;
    } else {
        dotPos = (result.spoof_prob / thresh) * 50;
    }
    // Constrain bounds
    dotPos = Math.max(0, Math.min(100, dotPos));
    
    // Delay score dot movement for smooth animation
    setTimeout(() => {
        scoreDot.style.left = `${dotPos}%`;
    }, 150);
    
    // 5. Update Heatmap visualization
    const heatmapCard = document.getElementById('heatmap-card');
    const heatmapImg = document.getElementById('heatmap-img');
    if (heatmapCard && heatmapImg) {
        if (result.heatmap_url) {
            heatmapImg.src = result.heatmap_url + '?t=' + new Date().getTime();
            heatmapCard.style.display = 'flex';
        } else {
            heatmapCard.style.display = 'none';
            heatmapImg.src = '';
        }
    }
}

