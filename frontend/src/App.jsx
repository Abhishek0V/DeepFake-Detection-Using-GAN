import { useState, useCallback } from 'react';
import UploadPanel from './components/UploadPanel';
import ResultsPanel from './components/ResultsPanel';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioSrc, setAudioSrc] = useState('');
  const [resultsState, setResultsState] = useState('empty'); // 'empty' | 'loading' | 'results'
  const [result, setResult] = useState(null);

  // Called when user picks or records a file
  const handleFileSelected = useCallback((file) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setAudioSrc(url);
  }, []);

  // Reset everything
  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setAudioSrc('');
    setResultsState('empty');
    setResult(null);
  }, []);

  // Send to Flask /predict
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setResultsState('loading');
    setResult(null);

    const formData = new FormData();
    formData.append('audio', selectedFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Server error occurred');
      }

      const data = await response.json();
      if (data.success) {
        setResult(data);
        setResultsState('results');
      } else {
        throw new Error('Analysis request failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Analysis Error: ' + error.message);
      setResultsState('empty');
    }
  };

  return (
    <>
      {/* Ambient glow background */}
      <div className="glow-bg"></div>

      <div className="container">
        {/* ── Header ── */}
        <header>
          <div className="brand">
            <i className="fa-solid fa-shield-halved logo-icon"></i>
            <div>
              <h1>
                VoiceGuard <span className="accent">AI</span>
              </h1>
              <p className="tagline">Audio Anti-Spoofing &amp; Deepfake Detection</p>
            </div>
          </div>
        </header>

        {/* ── Main Grid ── */}
        <main className="grid-container">
          {/* Left: Upload / Record + Analyze button */}
          <UploadPanel
            onFileSelected={handleFileSelected}
            selectedFile={selectedFile}
            onReset={handleReset}
            onAnalyze={handleAnalyze}
            isAnalyzing={resultsState === 'loading'}
          />

          {/* Right: Results */}
          <ResultsPanel
            state={resultsState}
            result={result}
            audioSrc={audioSrc}
          />
        </main>

        {/* ── Footer ── */}
        <footer>
          <p>
            VoiceGuard AI &copy; 2026. Powered by PyTorch &amp; AASIST-L Anti-Spoofing
            Architecture (EER 2.74%).
          </p>
        </footer>
      </div>
    </>
  );
}
