import { useState, useRef, useCallback } from 'react';
import { useRecorder } from '../hooks/useRecorder';

export default function UploadPanel({ onFileSelected, selectedFile, onReset, onAnalyze, isAnalyzing }) {
  const [activeTab, setActiveTab] = useState('upload');
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordStatus, setRecordStatus] = useState('Click to start recording');
  const [timerText, setTimerText] = useState('00:00');
  const fileInputRef = useRef(null);

  const handleFileReady = useCallback(
    (file) => {
      onFileSelected(file);
      setRecordStatus('Recording saved. Ready for analysis.');
    },
    [onFileSelected]
  );

  const { canvasRef, startRecording, stopRecording } = useRecorder({
    onFileReady: handleFileReady,
  });

  // ── File Handling ──────────────────────────────────────────────
  const handleFile = useCallback(
    (file) => {
      if (!file.type.startsWith('audio/')) {
        alert('Please select an audio file (WAV, FLAC, MP3, etc.).');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const onDragEnter = (e) => { e.preventDefault(); setIsDragOver(true); };
  const onDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
  const onDragLeave = (e) => { e.preventDefault(); setIsDragOver(false); };
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  };
  const onFileInputChange = (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  };

  // ── Tab Switching ──────────────────────────────────────────────
  const switchTab = (tab) => {
    setActiveTab(tab);
    if (tab === 'upload' && isRecording) {
      handleStopRecording();
    }
  };

  // ── Recording ──────────────────────────────────────────────────
  const handleToggleRecording = () => {
    if (isRecording) {
      handleStopRecording();
    } else {
      handleStartRecording();
    }
  };

  const handleStartRecording = () => {
    setIsRecording(true);
    setRecordStatus('Recording live...');
    setTimerText('00:00');
    startRecording((event, value) => {
      if (event === 'timer') setTimerText(value);
      if (event === 'idle') { setIsRecording(false); setRecordStatus('Click to start recording'); }
    });
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    stopRecording((event) => {
      if (event === 'saved') setRecordStatus('Recording saved. Ready for analysis.');
      if (event === 'idle') setRecordStatus('Click to start recording');
    });
  };

  return (
    <section className="panel upload-section">
      <h2>
        <i className="fa-solid fa-microphone-lines section-icon"></i>
        Audio Analyzer
      </h2>
      <p className="description">
        Upload an audio file or record directly using your microphone. Our AASIST-L
        deep learning model will run structural analysis on the waveform.
      </p>

      {/* Tabs */}
      <div className="tab-container">
        <button
          className={`tab-btn${activeTab === 'upload' ? ' active' : ''}`}
          onClick={() => switchTab('upload')}
        >
          <i className="fa-solid fa-cloud-arrow-up"></i> Upload File
        </button>
        <button
          className={`tab-btn${activeTab === 'record' ? ' active' : ''}`}
          onClick={() => switchTab('record')}
        >
          <i className="fa-solid fa-microphone"></i> Record Live
        </button>
      </div>

      {/* Tab 1: Dropzone */}
      <div className={`tab-content${activeTab === 'upload' ? ' active' : ''}`}>
        <div
          className={`dropzone${isDragOver ? ' dragover' : ''}`}
          onDragEnter={onDragEnter}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="file-input"
            onChange={onFileInputChange}
          />
          <i className="fa-solid fa-arrow-up-from-bracket upload-icon"></i>
          <h3>Drag &amp; drop audio file here</h3>
          <p className="upload-hint">Supports WAV, FLAC, MP3, etc. (Auto-resampled to 16kHz)</p>
          <button
            className="btn btn-secondary"
            onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
          >
            Browse Files
          </button>
        </div>
      </div>

      {/* Tab 2: Recorder */}
      <div className={`tab-content${activeTab === 'record' ? ' active' : ''}`}>
        <div className="recorder-container">
          <canvas ref={canvasRef} className="visualizer"></canvas>
          <div className="recorder-timer">{timerText}</div>
          <div className="recorder-controls">
            <button
              className={`record-btn${isRecording ? ' recording' : ''}`}
              onClick={handleToggleRecording}
            >
              <i
                className={`fa-solid ${isRecording ? 'fa-square' : 'fa-circle'} icon-record`}
              ></i>
            </button>
            <p className="record-status">{recordStatus}</p>
          </div>
        </div>
      </div>

      {/* Selected File Card */}
      {selectedFile && (
        <div className="selected-file-card">
          <div className="file-info">
            <i className="fa-solid fa-file-audio file-icon"></i>
            <div className="file-details">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </span>
            </div>
          </div>
          <button className="btn-remove" onClick={onReset}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>
      )}

      {/* Analyze Button */}
      <button
        className="btn btn-primary btn-block"
        disabled={!selectedFile || isAnalyzing}
        onClick={onAnalyze}
      >
        <i className="fa-solid fa-wand-magic-sparkles"></i>
        {isAnalyzing ? 'Analyzing…' : 'Run Anti-Spoofing Analysis'}
      </button>
    </section>
  );
}
