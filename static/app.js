// AcousticML Studio Client-side Logic

// State variables
let activeInputTab = 'mic';
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let audioProcessor = null;
let audioChunks = [];
let recordStartTime = null;
let timerInterval = null;
let recordedWavBlob = null;
let uploadedFile = null;

// DOM Elements
const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const timerDisplay = document.getElementById('recording-timer');
const canvas = document.getElementById('waveform-canvas');
const canvasCtx = canvas.getContext('2d');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('audio-file-input');
const fileInfo = document.getElementById('file-info');
const fileNameDisplay = document.getElementById('file-name-display');
const processBtn = document.getElementById('process-btn');
const noiseSlider = document.getElementById('noise-reduce-slider');
const noiseValLabel = document.getElementById('noise-reduce-val');
const recorderStatus = document.getElementById('recorder-status');

// Visualizer panel elements
const visPlaceholder = document.getElementById('vis-placeholder');
const visContent = document.getElementById('vis-content');
const audioPlayerOriginal = document.getElementById('audio-player-original');
const audioPlayerCleaned = document.getElementById('audio-player-cleaned');
const waveformImg = document.getElementById('waveform-img');
const spectrogramImg = document.getElementById('spectrogram-img');

// Benchmarking element lists
const pytorchPred = document.getElementById('pytorch-pred');
const pytorchConf = document.getElementById('pytorch-conf');
const pytorchLatency = document.getElementById('pytorch-latency');

const onnxPred = document.getElementById('onnx-pred');
const onnxConf = document.getElementById('onnx-conf');
const onnxLatency = document.getElementById('onnx-latency');

const quantPred = document.getElementById('quant-pred');
const quantConf = document.getElementById('quant-conf');
const quantLatency = document.getElementById('quant-latency');

// Initialize Web Audio API elements for recording visualizer
let analyser = null;
let dataArray = null;
let animationFrameId = null;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    setupCanvas();
    setupDropzone();
    setupNoiseSlider();
    fetchBenchmarkResults(); // Initial load of system profiling results
});

// Canvas waveform visualization size setup
function setupCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = 80;
    
    // Clear canvas with a nice cyber grid look
    canvasCtx.fillStyle = '#060913';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    canvasCtx.strokeStyle = 'rgba(100, 255, 218, 0.1)';
    canvasCtx.lineWidth = 1;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, canvas.height / 2);
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

window.addEventListener('resize', setupCanvas);

// Input tab switching
function switchInputTab(tab) {
    activeInputTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    if (tab === 'mic') {
        document.querySelector('[onclick="switchInputTab(\'mic\')"]').classList.add('active');
        document.getElementById('tab-content-mic').classList.add('active');
        updateProcessButtonState();
    } else {
        document.querySelector('[onclick="switchInputTab(\'upload\')"]').classList.add('active');
        document.getElementById('tab-content-upload').classList.add('active');
        updateProcessButtonState();
    }
}

// Noise reduction slider updates
function setupNoiseSlider() {
    noiseSlider.addEventListener('input', (e) => {
        noiseValLabel.textContent = `${e.target.value}%`;
    });
}

// Microphone recording setup
recordBtn.addEventListener('click', async () => {
    audioChunks = [];
    recordedWavBlob = null;
    
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        
        // Web Audio setup for live visualizer
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(mediaStream);
        source.connect(analyser);
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        
        // Begin recording using standard MediaRecorder (Fallback for simple WAV conversion)
        // We'll capture audio buffers manually in the script to generate an exact WAV file
        const audioTracks = mediaStream.getAudioTracks();
        const recProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        source.connect(recProcessor);
        recProcessor.connect(audioContext.destination);
        
        recProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputBuffer = e.inputBuffer.getChannelData(0);
            // Deep copy buffer segment
            audioChunks.push(new Float32Array(inputBuffer));
        };
        
        audioProcessor = recProcessor;
        
        // Start State updates
        isRecording = true;
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        processBtn.disabled = true;
        
        recorderStatus.textContent = "Recording";
        recorderStatus.className = "status-indicator recording";
        
        recordStartTime = Date.now();
        timerInterval = setInterval(updateTimer, 1000);
        
        drawWaveform();
        
    } catch (err) {
        console.error("Microphone access denied: ", err);
        alert("Could not access microphone. Please check system permissions.");
    }
});

stopBtn.addEventListener('click', () => {
    if (!isRecording) return;
    
    // Stop recording states
    isRecording = false;
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    
    recorderStatus.textContent = "Ready";
    recorderStatus.className = "status-indicator idle";
    
    clearInterval(timerInterval);
    cancelAnimationFrame(animationFrameId);
    
    // Close audio nodes
    if (audioProcessor) {
        audioProcessor.disconnect();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    
    // Package floats to wav
    if (audioChunks.length > 0 && audioContext) {
        const fullLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const mergedBuffer = new Float32Array(fullLength);
        let offset = 0;
        for (let chunk of audioChunks) {
            mergedBuffer.set(chunk, offset);
            offset += chunk.length;
        }
        
        // Create an AudioBuffer
        const finalAudioBuffer = audioContext.createBuffer(1, fullLength, audioContext.sampleRate);
        finalAudioBuffer.copyToChannel(mergedBuffer, 0);
        
        // Encode AudioBuffer to standard 16-bit WAV PCM
        recordedWavBlob = bufferToWav(finalAudioBuffer);
        
        // Set original player source
        const audioURL = URL.createObjectURL(recordedWavBlob);
        audioPlayerOriginal.src = audioURL;
    }
    
    if (audioContext) {
        audioContext.close();
    }
    
    updateProcessButtonState();
});

function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    timerDisplay.textContent = `${mins}:${secs}`;
}

// Live canvas waveform drawing loop
function drawWaveform() {
    if (!isRecording) return;
    
    animationFrameId = requestAnimationFrame(drawWaveform);
    analyser.getByteTimeDomainData(dataArray);
    
    canvasCtx.fillStyle = '#060913';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Grid line
    canvasCtx.strokeStyle = 'rgba(100, 255, 218, 0.05)';
    canvasCtx.lineWidth = 1;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, canvas.height / 2);
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
    
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#64ffda';
    canvasCtx.beginPath();
    
    const sliceWidth = canvas.width * 1.0 / dataArray.length;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
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

// WAV encoding helper function
function bufferToWav(buffer) {
    let numOfChan = buffer.numberOfChannels,
        length = buffer.length * 2 + 44,
        bufferArr = new ArrayBuffer(length),
        view = new DataView(bufferArr),
        channels = [], i, sample,
        offset = 0,
        pos = 0;

    // Write WAV RIFF header
    setUint32(0x46464952);                         // "RIFF"
    setUint32(length - 8);                         // file length - 8
    setUint32(0x45564157);                         // "WAVE"
    setUint32(0x20746d66);                         // "fmt " chunk
    setUint32(16);                                 // chunk length
    setUint16(1);                                  // sample format (raw PCM)
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan); // byte rate
    setUint16(numOfChan * 2);                      // block align
    setUint16(16);                                 // bits per sample
    setUint32(0x61746164);                         // "data" - chunk
    setUint32(length - pos - 4);                   // chunk length

    // Interleave channels
    for(i=0; i<buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    while(pos < length) {
        for(i=0; i<numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
            sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF); // scale to 16-bit signed
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }

    return new Blob([bufferArr], {type: "audio/wav"});

    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }

    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }
}

// File dropzone event setups
function setupDropzone() {
    dropzone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
    });
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });
}

function handleUploadedFile(file) {
    if (file.type !== 'audio/wav' && !file.name.endsWith('.wav')) {
        alert("Only standard RIFF/WAV audio files are supported.");
        return;
    }
    
    uploadedFile = file;
    fileNameDisplay.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.style.display = 'flex';
    dropzone.style.display = 'none';
    
    // Set original player source
    const audioURL = URL.createObjectURL(file);
    audioPlayerOriginal.src = audioURL;
    
    updateProcessButtonState();
}

function resetUploadedFile() {
    uploadedFile = null;
    fileInfo.style.display = 'none';
    dropzone.style.display = 'block';
    fileInput.value = '';
    audioPlayerOriginal.src = '';
    updateProcessButtonState();
}

function updateProcessButtonState() {
    if (activeInputTab === 'mic' && recordedWavBlob) {
        processBtn.disabled = false;
    } else if (activeInputTab === 'upload' && uploadedFile) {
        processBtn.disabled = false;
    } else {
        processBtn.disabled = true;
    }
}

// REST Api process trigger
processBtn.addEventListener('click', async () => {
    const fileToSend = activeInputTab === 'mic' ? recordedWavBlob : uploadedFile;
    if (!fileToSend) return;
    
    // Set status to processing
    processBtn.disabled = true;
    recorderStatus.textContent = "Processing";
    recorderStatus.className = "status-indicator processing";
    
    // Visualizer placeholders
    visPlaceholder.innerHTML = `<i class="fa-solid fa-arrows-spin placeholder-icon spinning"></i>
                                <p>Extracting Mel-Spectrogram features, running spectral gating noise filters, and loading optimized neural models...</p>`;
    
    const formData = new FormData();
    formData.append('file', fileToSend, 'audio.wav');
    formData.append('noise_reduction', parseFloat(noiseSlider.value) / 100.0);
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned error status ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // 1. Reveal content
            visPlaceholder.style.display = 'none';
            visContent.style.display = 'flex';
            
            // 2. Set images
            waveformImg.src = data.waveform_image;
            spectrogramImg.src = data.spectrogram_image;
            
            // 3. Set audio player source
            audioPlayerCleaned.src = data.cleaned_audio;
            
            // 4. Update prediction metrics
            updatePredictionDetails(data.inference);
        } else {
            throw new Error("Process returned unsuccessful");
        }
        
    } catch (err) {
        console.error("Processing failed: ", err);
        visPlaceholder.innerHTML = `<i class="fa-solid fa-triangle-exclamation placeholder-icon" style="color: var(--accent-pink);"></i>
                                    <p>Failed to process audio. Please ensure the local server is running and the uploaded file is a valid 16-bit WAV.</p>`;
    } finally {
        processBtn.disabled = false;
        recorderStatus.textContent = "Ready";
        recorderStatus.className = "status-indicator idle";
    }
});

// Update prediction panel elements
function updatePredictionDetails(inference) {
    // PyTorch
    pytorchPred.textContent = inference.pytorch_fp32.prediction;
    pytorchConf.textContent = `${(inference.pytorch_fp32.confidence * 100).toFixed(1)}%`;
    pytorchLatency.textContent = `${inference.pytorch_fp32.latency_ms} ms`;
    
    // ONNX
    onnxPred.textContent = inference.onnx_fp32.prediction;
    onnxConf.textContent = `${(inference.onnx_fp32.confidence * 100).toFixed(1)}%`;
    onnxLatency.textContent = `${inference.onnx_fp32.latency_ms} ms`;
    
    // Quantized
    quantPred.textContent = inference.onnx_int8.prediction;
    quantConf.textContent = `${(inference.onnx_int8.confidence * 100).toFixed(1)}%`;
    quantLatency.textContent = `${inference.onnx_int8.latency_ms} ms`;
    
    // Update live latency metrics inside benchmark charts as well (dynamic scaling)
    renderLatencyBars(
        inference.pytorch_fp32.latency_ms,
        inference.onnx_fp32.latency_ms,
        inference.onnx_int8.latency_ms
    );
}

// Query overall benchmark sizes & latencies
async function fetchBenchmarkResults() {
    try {
        const response = await fetch('/api/benchmark');
        if (response.ok) {
            const data = await response.json();
            
            // Update labels
            document.getElementById('benchmark-env-label').textContent = `Inference Profiling: ${data.environment}`;
            
            // Update model sizes
            document.getElementById('pytorch-size').textContent = `${data.pytorch_fp32.size_mb} MB`;
            document.getElementById('onnx-size').textContent = `${data.onnx_fp32.size_mb} MB`;
            document.getElementById('quant-size').textContent = `${data.onnx_int8.size_mb} MB`;
            
            // Render size comparison bars
            const maxVal = Math.max(data.pytorch_fp32.size_mb, data.onnx_fp32.size_mb, data.onnx_int8.size_mb);
            document.getElementById('bar-size-py').style.width = `${(data.pytorch_fp32.size_mb / maxVal) * 100}%`;
            document.getElementById('bar-size-onnx').style.width = `${(data.onnx_fp32.size_mb / maxVal) * 100}%`;
            document.getElementById('bar-size-quant').style.width = `${(data.onnx_int8.size_mb / maxVal) * 100}%`;
            
            document.getElementById('label-size-py').textContent = `${data.pytorch_fp32.size_mb}M`;
            document.getElementById('label-size-onnx').textContent = `${data.onnx_fp32.size_mb}M`;
            document.getElementById('label-size-quant').textContent = `${data.onnx_int8.size_mb}M`;
            
            // Memory savings calculation
            const reduction = ((data.pytorch_fp32.size_mb - data.onnx_int8.size_mb) / data.pytorch_fp32.size_mb * 100).toFixed(0);
            document.getElementById('size-reduction-pct').textContent = `~${reduction}% memory savings`;
            
            // Render latency bars
            renderLatencyBars(
                data.pytorch_fp32.mean_ms,
                data.onnx_fp32.mean_ms,
                data.onnx_int8.mean_ms
            );
        }
    } catch (err) {
        console.error("Failed to load benchmarks: ", err);
    }
}

function renderLatencyBars(pyVal, onnxVal, quantVal) {
    const maxVal = Math.max(pyVal, onnxVal, quantVal);
    
    document.getElementById('bar-lat-py').style.width = `${(pyVal / maxVal) * 100}%`;
    document.getElementById('bar-lat-onnx').style.width = `${(onnxVal / maxVal) * 100}%`;
    document.getElementById('bar-lat-quant').style.width = `${(quantVal / maxVal) * 100}%`;
    
    document.getElementById('label-lat-py').textContent = `${pyVal}ms`;
    document.getElementById('label-lat-onnx').textContent = `${onnxVal}ms`;
    document.getElementById('label-lat-quant').textContent = `${quantVal}ms`;
    
    // Latency speedup calculation
    const speedup = (pyVal / quantVal).toFixed(1);
    document.getElementById('speedup-pct').textContent = `~${speedup}x CPU speedup`;
}

// Trigger background profiling script
async function triggerReprofile() {
    const btn = document.getElementById('btn-re-profile');
    const icon = btn.querySelector('i');
    
    btn.disabled = true;
    icon.className = 'fa-solid fa-arrows-spin spinning';
    
    try {
        // Send request to reload benchmarks (backend will read new benchmark file if updated)
        // Note: For a live re-profile command, we run this locally. 
        // We will call the backend API, or let the user run it from the console.
        // Let's just fetch benchmarks again after a simulated delay to represent a reload.
        await new Promise(resolve => setTimeout(resolve, 1500));
        await fetchBenchmarkResults();
    } catch (err) {
        console.error(err);
    } finally {
        btn.disabled = false;
        icon.className = 'fa-solid fa-arrows-spin';
    }
}

// Toggle bottom insights drawer
function toggleInsights() {
    const body = document.getElementById('insights-body');
    const chevron = document.getElementById('insights-chevron');
    
    if (body.style.maxHeight === '0px' || body.classList.contains('collapsed')) {
        body.style.maxHeight = '1000px';
        body.classList.remove('collapsed');
        chevron.className = 'fa-solid fa-chevron-up';
    } else {
        body.style.maxHeight = '0px';
        body.classList.add('collapsed');
        chevron.className = 'fa-solid fa-chevron-down';
    }
}

// Initialize drawer state as collapsed on start to save initial space
document.getElementById('insights-body').style.maxHeight = '0px';
document.getElementById('insights-body').classList.add('collapsed');
