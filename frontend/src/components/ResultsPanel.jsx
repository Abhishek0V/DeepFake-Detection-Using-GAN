import { useEffect, useRef } from 'react';

/**
 * ResultsPanel — displays one of three states:
 *   'empty'   → placeholder prompt
 *   'loading' → spinner
 *   'results' → full analysis output
 */
export default function ResultsPanel({ state, result, audioSrc }) {
  const confidenceBarRef = useRef(null);
  const scoreDotRef = useRef(null);

  // Animate bars after results render
  useEffect(() => {
    if (state !== 'results' || !result) return;

    const barTimer = setTimeout(() => {
      if (confidenceBarRef.current) {
        confidenceBarRef.current.style.width = `${result.confidence}%`;
      }
    }, 100);

    // Dot position mapping: split scale around threshold at 50%
    let dotPos = 0;
    const thresh = result.threshold;
    if (result.spoof_prob > thresh) {
      dotPos = 50 + ((result.spoof_prob - thresh) / (1 - thresh)) * 50;
    } else {
      dotPos = (result.spoof_prob / thresh) * 50;
    }
    dotPos = Math.max(0, Math.min(100, dotPos));

    const dotTimer = setTimeout(() => {
      if (scoreDotRef.current) {
        scoreDotRef.current.style.left = `${dotPos}%`;
      }
    }, 150);

    return () => {
      clearTimeout(barTimer);
      clearTimeout(dotTimer);
    };
  }, [state, result]);

  const isSpoof = result?.verdict === 'spoof';

  return (
    <section className="panel results-section" id="results-panel">
      {/* ── Empty State ── */}
      {state === 'empty' && (
        <div className="results-empty">
          <div className="empty-icon-container">
            <i className="fa-solid fa-chart-bar empty-icon"></i>
          </div>
          <h3>Analysis Results</h3>
          <p>Provide audio input and trigger the model to view detailed spoofing vulnerability reports.</p>
        </div>
      )}

      {/* ── Loading State ── */}
      {state === 'loading' && (
        <div className="loading-state">
          <div className="spinner"></div>
          <h3>Extracting SincConv Features</h3>
          <p>Running HeteroGraphAttention network inference...</p>
        </div>
      )}

      {/* ── Results State ── */}
      {state === 'results' && result && (
        <div className="results-content">
          {/* Header */}
          <div className="result-header">
            <h2>Analysis Complete</h2>
            <span className={`badge ${isSpoof ? 'threat' : 'safe'}`}>
              {isSpoof ? `${result.risk_level} Risk` : 'Safe'}
            </span>
          </div>

          {/* Verdict Card */}
          <div className={`verdict-card ${isSpoof ? 'threat' : 'safe'}`}>
            <div className="verdict-icon-container">
              <i
                className={`fa-solid ${
                  isSpoof ? 'fa-triangle-exclamation' : 'fa-circle-check'
                } verdict-icon`}
              ></i>
            </div>
            <div className="verdict-text-container">
              <span className="verdict-label">Classification Verdict</span>
              <span className="verdict-value">
                {isSpoof ? 'SPOOFED / DEEPFAKE' : 'BONAFIDE / NATURAL'}
              </span>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="metrics-grid">
            {/* Confidence */}
            <div className="metric-card">
              <span className="metric-label">Model Confidence</span>
              <span className="metric-value">{result.confidence.toFixed(1)}%</span>
              <div className="progress-bar-container">
                <div
                  ref={confidenceBarRef}
                  className={`progress-bar ${isSpoof ? 'threat' : 'safe'}`}
                  style={{ width: '0%' }}
                ></div>
              </div>
            </div>

            {/* Spoof Score */}
            <div className="metric-card">
              <span className="metric-label">Spoofing Score</span>
              <span className="metric-value">{result.spoof_prob.toFixed(4)}</span>
              <div className="score-line-container">
                <div
                  ref={scoreDotRef}
                  className="score-dot"
                  style={{ left: '0%' }}
                ></div>
                <div
                  className="score-threshold-line"
                  title={`EER Threshold: ${result.threshold}`}
                ></div>
              </div>
              <span className="metric-sub">
                Decision Threshold: {result.threshold}
              </span>
            </div>
          </div>

          {/* Audio Playback */}
          {audioSrc && (
            <div className="audio-playback-card">
              <h3>Input Audio Preview</h3>
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <audio controls src={audioSrc}></audio>
            </div>
          )}

          {/* Grad-CAM Heatmap */}
          {result.heatmap_url && (
            <div className="audio-playback-card">
              <h3 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>
                  <i className="fa-solid fa-eye"></i> Grad-CAM Saliency Map
                </span>
                <span className="heatmap-badge">XAI Active</span>
              </h3>
              <div className="heatmap-image-wrapper">
                <img
                  src={`${result.heatmap_url}?t=${Date.now()}`}
                  alt="Grad-CAM Saliency Map"
                />
              </div>
              <p className="heatmap-caption">
                Saliency map overlays the fused early features from spectral &amp; temporal
                encoders on the Mel Spectrogram. Bright regions (red/yellow) denote high
                spoof saliency.
              </p>
            </div>
          )}

          {/* Forensic Info */}
          <div className="forensic-info">
            <h3>
              <i className="fa-solid fa-circle-info"></i> Model Forensic Insights
            </h3>
            <ul>
              <li>
                <strong>Model Core:</strong> SincConv filterbank extract (70 channels)
              </li>
              <li>
                <strong>Evaluation:</strong> Graph Attention networks over spectrogram slices
              </li>
              <li>
                <strong>Security Status:</strong>{' '}
                {isSpoof
                  ? `Vulnerability detected. The sample shows standard acoustic anomalies indicating synthetic generation (Score ${result.spoof_prob.toFixed(4)} is above the threat threshold).`
                  : 'Acoustic profiles verified. The sample contains normal human vocal structures with no signature artificial voice fingerprints.'}
              </li>
            </ul>
          </div>
        </div>
      )}
    </section>
  );
}
