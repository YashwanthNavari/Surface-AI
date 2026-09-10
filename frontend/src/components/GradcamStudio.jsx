import React, { useState, useEffect } from 'react';
import { Eye, Layers, Cpu, CheckCircle2, Sliders, Info, ArrowLeft } from 'lucide-react';

const MODELS = [
  { id: 'efficientnet_finetuned', name: 'Fine-Tuned EfficientNetB0', badge: 'Best Authority' },
  { id: 'efficientnet_frozen', name: 'EfficientNetB0 (Frozen)', badge: 'Transfer Learning' },
  { id: 'custom_cnn', name: 'Custom CNN (Scratch)', badge: 'From Scratch' },
];

export default function GradcamStudio({ currentFile, currentResult, onBackToConsole }) {
  const [selectedModel, setSelectedModel] = useState('efficientnet_finetuned');
  const [gradcamData, setGradcamData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

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
    return name.replace('_', ' ').replace('-', ' ').toUpperCase();
  };

  return (
    <div>
      <div className="card" style={{ marginBottom: '1.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
              <button className="btn-secondary" onClick={onBackToConsole} style={{ padding: '0.35rem 0.7rem' }}>
                <ArrowLeft className="w-4 h-4" /> Back
              </button>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                GRAD-CAM EXPLAINABILITY STUDIO
              </h2>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
              Investigate the spatial gradient activations across convolutional feature maps to verify defect localization.
            </p>
          </div>

          {/* Model Selector Pills */}
          <div className="model-pill-selector" style={{ margin: 0 }}>
            {MODELS.map((m) => (
              <button
                key={m.id}
                className={`pill-btn ${selectedModel === m.id ? 'active' : ''}`}
                onClick={() => setSelectedModel(m.id)}
              >
                {m.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="card" style={{ textAlign: 'center', padding: '5rem 2rem' }}>
          <div className="spinner" style={{ width: '40px', height: '40px', marginBottom: '1rem' }}></div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>COMPUTING GRADIENT FEATURE ACTIVATIONS...</div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            Extracting gradient flow with respect to target convolutional tensor
          </div>
        </div>
      ) : gradcamData ? (
        <div className="space-y-4">
          {/* Telemetry Bar */}
          <div className="card" style={{ padding: '1rem 1.4rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-elevated)' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                Target Architecture:
              </span>{' '}
              <strong style={{ color: 'var(--text-primary)' }}>{MODELS.find((m) => m.id === selectedModel)?.name}</strong>
            </div>

            <div style={{ display: 'flex', gap: '2rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Identified Class:
                </span>{' '}
                <strong style={{ color: 'var(--cyan-primary)' }}>{formatClassName(gradcamData.predicted_class)}</strong>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Target Confidence:
                </span>{' '}
                <strong style={{ color: 'var(--emerald-pass)', fontFamily: 'var(--font-mono)' }}>
                  {(gradcamData.confidence * 100).toFixed(2)}%
                </strong>
              </div>
            </div>
          </div>

          {/* 3-Panel Triplet Layout */}
          <div className="card">
            <div className="gradcam-layout">
              {/* Panel 1: Original Image */}
              <div className="gradcam-panel">
                <div className="gradcam-panel-title">
                  <Eye className="w-4 h-4 text-slate-400" />
                  1. Input Component Surface
                </div>
                <img src={gradcamData.original_base64} alt="Original input" className="gradcam-image" />
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.6rem' }}>
                  Raw 224 × 224 pixel matrix input
                </p>
              </div>

              {/* Panel 2: Class Activation Heatmap */}
              <div className="gradcam-panel">
                <div className="gradcam-panel-title">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  2. Grad-CAM Activation Heatmap
                </div>
                <img src={gradcamData.heatmap_base64} alt="Grad-CAM heatmap" className="gradcam-image" />
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.6rem' }}>
                  ReLU gradient importance map (Jet scale)
                </p>
              </div>

              {/* Panel 3: Superimposed Overlay */}
              <div className="gradcam-panel">
                <div className="gradcam-panel-title">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  3. Defect Localization Overlay
                </div>
                <img src={gradcamData.overlay_base64} alt="Superimposed overlay" className="gradcam-image" />
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.6rem' }}>
                  Superimposed attention on defect locus
                </p>
              </div>
            </div>

            {/* Explainability Discussion Card */}
            <div style={{ marginTop: '1.4rem', padding: '1rem 1.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', color: 'var(--text-primary)', fontSize: '0.88rem', fontWeight: 700 }}>
                <Info className="w-4 h-4 text-cyan-400" />
                Explainable AI (XAI) Academic Insight
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                Grad-CAM utilizes the gradient information flowing into the final convolutional layer to calculate visual importance weights for each feature map. High activation regions (represented by red and yellow peaks in the Jet colormap) demonstrate that the model successfully attends to the genuine physical defect morphology—such as scratch grooves or crack fissures—rather than spurious background illumination artifacts.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="card" style={{ textAlign: 'center', padding: '5rem 2rem' }}>
          <Eye className="w-14 h-14 text-slate-500" style={{ margin: '0 auto 1.2rem', opacity: 0.6 }} />
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>NO INSPECTION IMAGE LOADED</div>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)', margin: '0.6rem 0 1.2rem' }}>
            Please perform a defect inspection first in the Live Console to explore Grad-CAM visual heatmaps.
          </div>
          <button className="btn-primary" onClick={onBackToConsole}>
            Go to Live Console
          </button>
        </div>
      )}
    </div>
  );
}
