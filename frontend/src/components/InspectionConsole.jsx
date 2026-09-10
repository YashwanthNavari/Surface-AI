import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertTriangle, XCircle, ArrowRight, Eye, RefreshCw, Cpu, Layers } from 'lucide-react';

const DEFECT_CLASSES = [
  { id: 'crazing', name: 'Crazing', desc: 'High-density micro-fractures' },
  { id: 'inclusion', name: 'Inclusion', desc: 'Embedded particulate matter' },
  { id: 'patches', name: 'Patches', desc: 'Localized surface irregularities' },
  { id: 'pitted_surface', name: 'Pitted Surface', desc: 'Cavities & surface porosity' },
  { id: 'rolled-in_scale', name: 'Rolled-in Scale', desc: 'Compressed mill oxide defects' },
  { id: 'scratches', name: 'Scratches', desc: 'Directional abrasive grooves' },
];

export default function InspectionConsole({ onInspect, inspectionResult, loading, onOpenGradcam, onSampleSelect }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      processFile(file);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onInspect(file);
  };

  const handleSampleClick = (clsId) => {
    onSampleSelect(clsId, (file, url) => {
      setSelectedFile(file);
      setPreviewUrl(url);
    });
  };

  const clearSelection = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const formatClassName = (name) => {
    if (!name) return '—';
    return name.replace('_', ' ').replace('-', ' ').toUpperCase();
  };

  return (
    <div className="console-grid">
      {/* LEFT COLUMN: UPLOAD & QUICK SAMPLE PICKER */}
      <div className="space-y-4">
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <UploadCloud className="w-4 h-4 text-cyan-400" />
              Component Image Input
            </span>
            <span className="card-badge">224 × 224 RGB</span>
          </div>

          <div
            className={`dropzone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="file-input"
              accept="image/*"
              onChange={handleFileInput}
            />

            {previewUrl ? (
              <div className="preview-wrapper">
                <img src={previewUrl} alt="Component preview" className="preview-img" />
                <button className="remove-btn" onClick={clearSelection}>
                  Replace Image
                </button>
              </div>
            ) : (
              <div>
                <UploadCloud className="dropzone-icon" />
                <div className="dropzone-text-main">DRAG & DROP COMPONENT IMAGE</div>
                <div className="dropzone-text-sub">or click to browse local filesystem</div>
                <button
                  className="btn-secondary"
                  style={{ marginTop: '1rem' }}
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                >
                  Select File
                </button>
              </div>
            )}
          </div>

          {/* Quick Samples Section */}
          <div style={{ marginTop: '1.2rem' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.6rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Quick Load Benchmark Samples (NEU Test Set):
            </div>
            <div className="samples-grid">
              {DEFECT_CLASSES.map((cls) => (
                <div
                  key={cls.id}
                  className="sample-chip"
                  onClick={() => handleSampleClick(cls.id)}
                  title={cls.desc}
                >
                  <div className="sample-chip-name">{cls.name}</div>
                  <div className="sample-chip-sub">Test sample</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Engine Telemetry Card */}
        <div className="card" style={{ marginTop: '1.2rem' }}>
          <div className="card-header">
            <span className="card-title">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Inspection Architecture
            </span>
            <span className="card-badge">TensorFlow 2.21</span>
          </div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
              <span>Scratch Model:</span>
              <strong style={{ color: 'var(--text-primary)' }}>Custom 4-Stage CNN</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
              <span>Transfer Backbone:</span>
              <strong style={{ color: 'var(--text-primary)' }}>EfficientNetB0 (ImageNet)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Fine-Tuning:</span>
              <strong style={{ color: 'var(--cyan-primary)' }}>Top 30 Layers Unfrozen</strong>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT COLUMN: INSPECTION RESULT & CONSENSUS ARBITRATION */}
      <div>
        {loading ? (
          <div className="card" style={{ textAlign: 'center', padding: '5rem 2rem' }}>
            <div className="spinner" style={{ width: '40px', height: '40px', marginBottom: '1rem' }}></div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>ANALYZING SURFACE TOPOLOGY...</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
              Running inference across Custom CNN, EfficientNet Frozen, and Fine-Tuned models
            </div>
          </div>
        ) : inspectionResult ? (
          <div className="card inspection-hud">
            <div className="card-header">
              <span className="card-title">
                <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                Primary Inspection Telemetry
              </span>
              <span className="card-badge">
                ID #{inspectionResult.inspection_id || 'LOCAL'} • {inspectionResult.timestamp || 'REALTIME'}
              </span>
            </div>

            {/* Central Large Header */}
            <div className="hud-top">
              <div>
                <div className="primary-defect-label">Identified Defect Category</div>
                <div className="primary-defect-val">
                  {formatClassName(inspectionResult.primary_prediction)}
                </div>
              </div>
              <div className="primary-conf-badge">
                <div className="primary-conf-pct">
                  {(inspectionResult.primary_confidence * 100).toFixed(1)}%
                </div>
                <div className="primary-conf-lbl">Model Confidence</div>
              </div>
            </div>

            {/* Model Comparison Table */}
            <table className="model-comparison-hud">
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th style={{ textAlign: 'right' }}>Confidence Scale</th>
                </tr>
              </thead>
              <tbody>
                {/* Custom CNN */}
                {inspectionResult.models?.custom_cnn && (
                  <tr>
                    <td>
                      <div className="model-name-cell">
                        <Cpu className="w-4 h-4 text-indigo-400" />
                        Custom CNN (Scratch)
                      </div>
                    </td>
                    <td className="model-class-cell">
                      {formatClassName(inspectionResult.models.custom_cnn.predicted_class)}
                    </td>
                    <td className="model-conf-cell">
                      {(inspectionResult.models.custom_cnn.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="conf-bar-wrapper">
                        <div
                          className="conf-bar-fill"
                          style={{ width: `${inspectionResult.models.custom_cnn.confidence * 100}%` }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                )}

                {/* EfficientNet Frozen */}
                {inspectionResult.models?.efficientnet_frozen && (
                  <tr>
                    <td>
                      <div className="model-name-cell">
                        <Layers className="w-4 h-4 text-blue-400" />
                        EfficientNetB0 (Frozen)
                      </div>
                    </td>
                    <td className="model-class-cell">
                      {formatClassName(inspectionResult.models.efficientnet_frozen.predicted_class)}
                    </td>
                    <td className="model-conf-cell">
                      {(inspectionResult.models.efficientnet_frozen.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="conf-bar-wrapper">
                        <div
                          className="conf-bar-fill"
                          style={{ width: `${inspectionResult.models.efficientnet_frozen.confidence * 100}%` }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                )}

                {/* EfficientNet Fine-Tuned */}
                {inspectionResult.models?.efficientnet_finetuned && (
                  <tr>
                    <td>
                      <div className="model-name-cell">
                        <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                        Fine-Tuned EfficientNet
                        <span className="badge-champion">Top Authority</span>
                      </div>
                    </td>
                    <td className="model-class-cell" style={{ color: 'var(--emerald-pass)' }}>
                      {formatClassName(inspectionResult.models.efficientnet_finetuned.predicted_class)}
                    </td>
                    <td className="model-conf-cell" style={{ color: 'var(--emerald-pass)' }}>
                      {(inspectionResult.models.efficientnet_finetuned.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="conf-bar-wrapper">
                        <div
                          className="conf-bar-fill"
                          style={{
                            width: `${inspectionResult.models.efficientnet_finetuned.confidence * 100}%`,
                            background: 'linear-gradient(90deg, #10B981, #06B6D4)',
                          }}
                        ></div>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>

            {/* Consensus Arbitration Banner */}
            <div
              className={`consensus-banner ${
                inspectionResult.consensus_status === 'CONSENSUS_AGREED' ? 'agreed' : 'disagreed'
              }`}
            >
              <div className="consensus-left">
                {inspectionResult.consensus_status === 'CONSENSUS_AGREED' ? (
                  <CheckCircle2 className="consensus-icon text-emerald-400" />
                ) : (
                  <AlertTriangle className="consensus-icon text-amber-400" />
                )}
                <div>
                  <div className="consensus-title">
                    {inspectionResult.consensus_status === 'CONSENSUS_AGREED'
                      ? '✓ All Models Agree'
                      : '⚠ Model Disagreement Detected'}
                  </div>
                  <div className="consensus-subtitle">
                    Recommendation: <strong>{inspectionResult.recommendation}</strong>
                  </div>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div className="consensus-badge">
                  CONSENSUS: {inspectionResult.consensus_count} / {inspectionResult.total_models}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {inspectionResult.confidence_tier}
                </div>
              </div>
            </div>

            {/* Action Bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.2rem' }}>
              <button
                className="btn-primary"
                onClick={() => onOpenGradcam(selectedFile)}
              >
                <Eye className="w-4 h-4" />
                View Grad-CAM Explainability
              </button>

              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Verification protocol: <strong>NEU-CLS 6-Class Benchmark</strong>
              </div>
            </div>
          </div>
        ) : (
          /* Empty Initial State */
          <div className="card" style={{ textAlign: 'center', padding: '5rem 2rem' }}>
            <Cpu className="w-14 h-14 text-cyan-500" style={{ margin: '0 auto 1.2rem', opacity: 0.8 }} />
            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '0.5px' }}>
              READY FOR SURFACE INSPECTION
            </div>
            <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', maxWidth: '440px', margin: '0.6rem auto 1.5rem', lineHeight: 1.6 }}>
              Select an image from the test set chips on the left or upload any industrial component image to begin automated defect analysis.
            </div>
            <div style={{ display: 'inline-flex', gap: '0.8rem' }}>
              <button className="btn-secondary" onClick={() => handleSampleClick('scratches')}>
                Test "Scratches" Sample
              </button>
              <button className="btn-secondary" onClick={() => handleSampleClick('crazing')}>
                Test "Crazing" Sample
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
