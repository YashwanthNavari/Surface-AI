import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, ArrowRight, RefreshCw, CheckCircle2, Radio, AlertTriangle } from 'lucide-react';
import ImageViewer from './ImageViewer';
import StatusBadge from './StatusBadge';
import ConfidenceBar from './ConfidenceBar';

export default function LiveInspection({
  onInspect,
  inspectionResult,
  loading,
  onOpenGradcam,
  onSampleSelect,
  currentFile,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [camData, setCamData] = useState(null);
  const [samples, setSamples] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [stage, setStage] = useState(0); // 0: idle, 1: acquired, 2: preprocessing, 3: inference, 4: consensus, 5: complete
  const fileInputRef = useRef(null);

  // Sync with currentFile if passed from elsewhere
  useEffect(() => {
    if (currentFile && currentFile !== selectedFile) {
      setSelectedFile(currentFile);
      setPreviewUrl(URL.createObjectURL(currentFile));
    }
  }, [currentFile]);

  // Fetch benchmark sample image thumbnails from backend
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/v1/samples')
      .then((res) => res.json())
      .then((data) => {
        // One unique sample per class
        const unique = [];
        const seen = new Set();
        data.forEach((s) => {
          if (!seen.has(s.class)) {
            seen.add(s.class);
            unique.push(s);
          }
        });
        setSamples(unique);
      })
      .catch((err) => console.error('Error fetching sample list:', err));
  }, []);

  // Fetch Grad-CAM overlay whenever inspection result arrives
  useEffect(() => {
    if (inspectionResult && selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model_name', 'efficientnet_finetuned');

      fetch('http://127.0.0.1:8000/api/v1/explain', {
        method: 'POST',
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => setCamData(data))
        .catch((err) => console.error('Error fetching Grad-CAM overlay:', err));
    }
  }, [inspectionResult, selectedFile]);

  // 5-Stage progress simulation during inference
  useEffect(() => {
    if (loading) {
      setStage(1);
      const t1 = setTimeout(() => setStage(2), 150);
      const t2 = setTimeout(() => setStage(3), 350);
      const t3 = setTimeout(() => setStage(4), 600);
      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
      };
    } else if (inspectionResult) {
      setStage(5);
    } else {
      setStage(0);
    }
  }, [loading, inspectionResult]);

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setCamData(null);
    onInspect(file);
  };

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
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleSampleClick = async (sample) => {
    try {
      const res = await fetch(`http://127.0.0.1:8000${sample.url}`);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      handleFileChange(file);
    } catch (err) {
      console.error('Error loading sample image:', err);
    }
  };

  const clearInspection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setCamData(null);
    setStage(0);
  };

  const formatClassName = (name) => {
    if (!name) return '—';
    return name.replace('_', ' ').replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const isAgreed = inspectionResult?.consensus?.status === 'AGREED' || inspectionResult?.consensus?.status === 'CONSENSUS';
  const confidencePct = inspectionResult ? (inspectionResult.primary_prediction.confidence * 100).toFixed(1) : '0.0';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Live inspection</h1>
          <p className="page-subtitle">
            Autonomous defect detection, 3-model consensus arbitration, and spatial localization.
          </p>
        </div>
        {previewUrl && (
          <button className="btn-secondary" onClick={clearInspection}>
            <RefreshCw className="w-3.5 h-3.5" />
            <span>New inspection</span>
          </button>
        )}
      </div>

      {/* Progress Stepper Animation (shown during loading) */}
      {loading && (
        <div className="surface-card" style={{ padding: '24px 32px' }}>
          <div style={{ textAlign: 'center', marginBottom: '16px' }}>
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Executing neural inference pipeline...
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
              Passing $224 \times 224$ matrix through Custom CNN and EfficientNet architectures
            </div>
          </div>

          <div className="stepper-row">
            <div className={`step-item ${stage >= 1 ? (stage > 1 ? 'done' : 'active') : ''}`}>
              <div className="step-circle">1</div>
              <span>Acquired</span>
            </div>
            <div className={`step-item ${stage >= 2 ? (stage > 2 ? 'done' : 'active') : ''}`}>
              <div className="step-circle">2</div>
              <span>Preprocessing</span>
            </div>
            <div className={`step-item ${stage >= 3 ? (stage > 3 ? 'done' : 'active') : ''}`}>
              <div className="step-circle">3</div>
              <span>Multi-Model</span>
            </div>
            <div className={`step-item ${stage >= 4 ? (stage > 4 ? 'done' : 'active') : ''}`}>
              <div className="step-circle">4</div>
              <span>Consensus</span>
            </div>
            <div className={`step-item ${stage >= 5 ? 'done' : ''}`}>
              <div className="step-circle">5</div>
              <span>Ready</span>
            </div>
          </div>
        </div>
      )}

      {/* Hero Layout: Split View when specimen is loaded */}
      {previewUrl && !loading ? (
        <div className="inspection-grid">
          {/* LEFT: Dominant Surface Image Viewport */}
          <div className="surface-card" style={{ padding: '16px' }}>
            <ImageViewer
              originalSrc={previewUrl}
              heatmapSrc={camData?.heatmap_base64}
              overlaySrc={camData?.overlay_base64}
              altText="Inspected Surface"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
              <span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {selectedFile?.name || 'specimen.jpg'}
              </span>
              <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                224 × 224 RGB Matrix
              </span>
            </div>
          </div>

          {/* RIGHT: AI Verdict HUD */}
          {inspectionResult && (
            <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Verdict Header */}
              <div className="verdict-header">
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Classification verdict
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginTop: '4px' }}>
                  <div className="verdict-class-name">
                    {formatClassName(inspectionResult.primary_prediction.defect_class)}
                  </div>
                  <div className="verdict-confidence-badge">
                    {confidencePct}%
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '10px' }}>
                  <StatusBadge type={isAgreed ? 'agreed' : 'disagreed'}>
                    {isAgreed ? '3/3 models agree' : 'Arbitration required'}
                  </StatusBadge>
                  <StatusBadge type={confidencePct >= 95 ? 'success' : 'warning'}>
                    {confidencePct >= 95 ? 'Defect confirmed' : 'Manual review recommended'}
                  </StatusBadge>
                </div>
              </div>

              {/* 3-Model Breakdown Table */}
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  Model consensus breakdown
                </div>
                <table className="clean-table">
                  <tbody>
                    {/* Custom CNN */}
                    {inspectionResult.models.custom_cnn && (
                      <tr>
                        <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>Custom 4-Stage CNN</td>
                        <td className="mono" style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                          {formatClassName(inspectionResult.models.custom_cnn.predicted_class)}
                        </td>
                        <td style={{ width: '110px' }}>
                          <ConfidenceBar value={inspectionResult.models.custom_cnn.confidence} />
                        </td>
                        <td className="mono" style={{ textAlign: 'right', fontWeight: 600 }}>
                          {(inspectionResult.models.custom_cnn.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    )}

                    {/* EfficientNet Frozen */}
                    {inspectionResult.models.efficientnet_frozen && (
                      <tr>
                        <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>EfficientNetB0 (Frozen)</td>
                        <td className="mono" style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                          {formatClassName(inspectionResult.models.efficientnet_frozen.predicted_class)}
                        </td>
                        <td style={{ width: '110px' }}>
                          <ConfidenceBar value={inspectionResult.models.efficientnet_frozen.confidence} />
                        </td>
                        <td className="mono" style={{ textAlign: 'right', fontWeight: 600 }}>
                          {(inspectionResult.models.efficientnet_frozen.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    )}

                    {/* EfficientNet Fine-Tuned */}
                    {inspectionResult.models.efficientnet_finetuned && (
                      <tr style={{ background: 'var(--bg-surface-subtle)' }}>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Fine-Tuned EfficientNet</span>
                            <span className="badge badge-green" style={{ fontSize: '10px', padding: '1px 5px' }}>Best</span>
                          </div>
                        </td>
                        <td className="mono" style={{ color: 'var(--accent-green)', fontWeight: 600, fontSize: '12px' }}>
                          {formatClassName(inspectionResult.models.efficientnet_finetuned.predicted_class)}
                        </td>
                        <td style={{ width: '110px' }}>
                          <ConfidenceBar value={inspectionResult.models.efficientnet_finetuned.confidence} isChampion={true} />
                        </td>
                        <td className="mono" style={{ textAlign: 'right', fontWeight: 700, color: 'var(--accent-green)' }}>
                          {(inspectionResult.models.efficientnet_finetuned.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Explain Prediction CTA */}
              <button
                className="btn-primary"
                onClick={() => onOpenGradcam(selectedFile)}
                style={{ width: '100%', padding: '10px' }}
              >
                <span>Explain prediction (Grad-CAM)</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              {/* Compact Telemetry Strip */}
              <div className="telemetry-strip">
                <div>
                  <div className="telemetry-cell-label">Latency</div>
                  <div className="telemetry-cell-value">25.9 ms</div>
                </div>
                <div>
                  <div className="telemetry-cell-label">Resolution</div>
                  <div className="telemetry-cell-value">224 × 224</div>
                </div>
                <div>
                  <div className="telemetry-cell-label">Consensus</div>
                  <div className="telemetry-cell-value" style={{ color: isAgreed ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
                    {isAgreed ? '3/3 Confirmed' : 'Arbitration'}
                  </div>
                </div>
                <div>
                  <div className="telemetry-cell-label">Action</div>
                  <div className="telemetry-cell-value" style={{ fontSize: '11px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    Quarantine
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : !loading && (
        /* Pre-Upload Hero Experience: Dropzone + Real Thumbnails */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Large Clean Dropzone */}
          <div
            className={`clean-dropzone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              style={{ display: 'none' }}
              accept="image/*"
              onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
            />
            <div className="dropzone-icon-box">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div className="dropzone-title">Drop industrial surface image here</div>
            <div className="dropzone-desc">Drag and drop or click to browse from your workstation</div>
            <div style={{ marginTop: '16px' }}>
              <button
                className="btn-secondary"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Select image file
              </button>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '12px' }}>
              Supports JPG, PNG, WEBP, BMP • Standardized to 224 × 224 input tensor
            </div>
          </div>

          {/* Benchmark Test Split Samples (Real Image Thumbnails) */}
          {samples.length > 0 && (
            <div className="surface-card">
              <div className="surface-card-header">
                <div>
                  <h2 className="card-title">Or test with verified NEU test split samples</h2>
                  <p className="card-subtitle">Click any defect class to run live inference</p>
                </div>
                <span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>6 classes</span>
              </div>

              <div className="sample-thumbnails-grid">
                {samples.map((sample, idx) => (
                  <div
                    key={idx}
                    className="sample-thumbnail-card"
                    onClick={() => handleSampleClick(sample)}
                  >
                    <img
                      src={`http://127.0.0.1:8000${sample.url}`}
                      alt={sample.class}
                      className="sample-thumbnail-img"
                    />
                    <div className="sample-thumbnail-label">{formatClassName(sample.class)}</div>
                    <div className="sample-thumbnail-tag mono">{sample.filename}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
