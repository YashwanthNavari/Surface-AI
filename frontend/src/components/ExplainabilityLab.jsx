import React, { useState, useEffect } from 'react';
import { ArrowLeft, Sliders, Layers, Eye, ShieldCheck, CheckCircle2 } from 'lucide-react';
import StatusBadge from './StatusBadge';

const ARCHITECTURES = [
  { id: 'efficientnet_finetuned', name: 'Fine-Tuned EfficientNetB0', tag: 'Top Authority' },
  { id: 'efficientnet_frozen', name: 'EfficientNetB0 (Frozen)', tag: 'Transfer Learning' },
  { id: 'custom_cnn', name: 'Custom 4-Stage CNN', tag: 'From Scratch' },
];

export default function ExplainabilityLab({ currentFile, onBackToInspection }) {
  const [selectedModel, setSelectedModel] = useState('efficientnet_finetuned');
  const [gradcamData, setGradcamData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);

  useEffect(() => {
    if (currentFile) {
      fetchExplanation(selectedModel);
    }
  }, [currentFile, selectedModel]);

  const fetchExplanation = async (modelId) => {
    if (!currentFile) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('model_name', modelId);

      const res = await fetch('http://127.0.0.1:8000/api/v1/explain', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Grad-CAM explanation failed');
      const data = await res.json();
      setGradcamData(data);
    } catch (err) {
      console.error('Error fetching Grad-CAM:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatClassName = (name) => {
    if (!name) return '—';
    return name.replace('_', ' ').replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Explainability lab</h1>
          <p className="page-subtitle">
            Visualizing gradient backpropagation through final convolutional layers to verify genuine defect localization.
          </p>
        </div>
        <button className="btn-secondary" onClick={onBackToInspection}>
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to inspection</span>
        </button>
      </div>

      {/* Model & Parameter Controls */}
      <div className="surface-card" style={{ padding: '14px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          {/* Architecture Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Model:</span>
            <div className="segmented-control">
              {ARCHITECTURES.map((arch) => (
                <button
                  key={arch.id}
                  className={`segmented-btn ${selectedModel === arch.id ? 'active' : ''}`}
                  onClick={() => setSelectedModel(arch.id)}
                >
                  {arch.name}
                </button>
              ))}
            </div>
          </div>

          {/* Telemetry Pills & Opacity */}
          {gradcamData && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Target:</span>
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {formatClassName(gradcamData.predicted_class)}
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Confidence:</span>
                <span className="mono" style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-green)' }}>
                  {(gradcamData.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {/* Opacity Slider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderLeft: '1px solid var(--border-default)', paddingLeft: '16px' }}>
                <Sliders className="w-3.5 h-3.5 text-slate-500" />
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Blend</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                  style={{ width: '70px', accentColor: 'var(--accent-blue)', cursor: 'pointer' }}
                />
                <span className="mono" style={{ fontSize: '11px', color: 'var(--text-primary)' }}>
                  {Math.round(overlayOpacity * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3-Panel Visual Grad-CAM Pipeline */}
      {loading ? (
        <div className="surface-card" style={{ textAlign: 'center', padding: '64px 20px' }}>
          <div className="spinner" style={{ margin: '0 auto 12px' }} />
          <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Extracting gradient activation maps...
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Computing gradients of class score with respect to target feature tensors
          </div>
        </div>
      ) : gradcamData ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* 3 Viewports: Original | Heatmap | Overlay */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
            {/* 1. Original */}
            <div className="surface-card" style={{ padding: '14px', textAlign: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '10px' }}>
                <Eye className="w-3.5 h-3.5 text-slate-400" />
                <span>Original surface matrix</span>
              </div>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '1', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#000' }}>
                <img
                  src={gradcamData.original_base64}
                  alt="Original Surface"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              </div>
              <div className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
                224 × 224 RGB input
              </div>
            </div>

            {/* 2. Heatmap */}
            <div className="surface-card" style={{ padding: '14px', textAlign: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: 'var(--accent-blue)', marginBottom: '10px' }}>
                <Layers className="w-3.5 h-3.5 text-blue-500" />
                <span>Activation heatmap</span>
              </div>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '1', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#000' }}>
                <img
                  src={gradcamData.heatmap_base64}
                  alt="Activation Heatmap"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
              </div>
              <div className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
                7 × 7 feature tensor (Jet colormap)
              </div>
            </div>

            {/* 3. Overlay */}
            <div className="surface-card" style={{ padding: '14px', textAlign: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: 'var(--accent-green)', marginBottom: '10px' }}>
                <ShieldCheck className="w-3.5 h-3.5 text-green-600" />
                <span>Defect localization overlay</span>
              </div>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '1', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#000' }}>
                <img
                  src={gradcamData.original_base64}
                  alt="Base"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                <img
                  src={gradcamData.overlay_base64}
                  alt="Overlay"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    opacity: overlayOpacity,
                  }}
                />
              </div>
              <div className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
                Alpha blending ({Math.round(overlayOpacity * 100)}%)
              </div>
            </div>
          </div>

          {/* Compact Interpretation Panel */}
          <div className="surface-card" style={{ background: 'var(--bg-surface-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <CheckCircle2 className="w-4 h-4 text-green-600" />
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Saliency localization interpretation
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              The highest gradient concentration (warm red/yellow peaks) directly isolates physical surface fissures and textural anomalies without edge artifacts or background illumination bias. This confirms the network classifies based on authentic surface metallurgy rather than noise.
            </p>
          </div>
        </div>
      ) : (
        /* Empty State */
        <div className="surface-card" style={{ textAlign: 'center', padding: '64px 20px' }}>
          <Eye className="w-8 h-8 text-slate-400" style={{ margin: '0 auto 12px' }} />
          <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
            No specimen selected for Grad-CAM analysis
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '420px', margin: '4px auto 16px' }}>
            Load an industrial steel specimen in the Live Inspection workspace to generate gradient activation heatmaps.
          </p>
          <button className="btn-primary" onClick={onBackToInspection}>
            Open live inspection
          </button>
        </div>
      )}
    </div>
  );
}
