import React, { useState, useEffect } from 'react';
import { BarChart3, Activity, Cpu, Layers, CheckCircle2, TrendingUp, Zap } from 'lucide-react';

export default function ResearchDashboard() {
  const [metricsData, setMetricsData] = useState({ models: [], per_class: {} });
  const [activeMatrixTab, setActiveMatrixTab] = useState('efficientnet_finetuned');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/v1/metrics')
      .then((res) => res.json())
      .then((data) => {
        setMetricsData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching metrics:', err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              ACADEMIC RESEARCH & EXPERIMENTAL BENCHMARKS
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
              Controlled evaluation across untouched test partition ($N = 270$ images, 45 per defect class).
            </p>
          </div>
          <span className="card-badge">EVALUATION METRIC: MACRO F1 & ACCURACY</span>
        </div>
      </div>

      {/* Top Stat Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        <div className="card" style={{ background: 'var(--bg-elevated)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
            Peak Classification Accuracy
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--emerald-pass)', marginTop: '0.4rem' }}>
            98.89%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            EfficientNetB0 (Fine-Tuned)
          </div>
        </div>

        <div className="card" style={{ background: 'var(--bg-elevated)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
            Scratch CNN Test Accuracy
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--cyan-primary)', marginTop: '0.4rem' }}>
            94.07%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Custom 4-Stage ConvNet
          </div>
        </div>

        <div className="card" style={{ background: 'var(--bg-elevated)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
            Transfer Learning Gain
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--indigo-accent)', marginTop: '0.4rem' }}>
            +4.82%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Over Scratch Architecture
          </div>
        </div>

        <div className="card" style={{ background: 'var(--bg-elevated)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
            Peak Inference Throughput
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--amber-warn)', marginTop: '0.4rem' }}>
            59.2 FPS
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Custom CNN (16.8 ms/image)
          </div>
        </div>
      </div>

      {/* Master Comparison Table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            Master Experimental Comparison Matrix
          </span>
          <span className="card-badge">results/model_comparison.csv</span>
        </div>

        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Paradigm</th>
                <th>Test Accuracy</th>
                <th>Macro Precision</th>
                <th>Macro Recall</th>
                <th>Macro F1-Score</th>
                <th>Latency (ms)</th>
                <th>FPS</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {/* Baseline */}
              <tr>
                <td>
                  <strong>HOG + GLCM + SVM</strong>
                </td>
                <td>Classical ML Baseline</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>92.22%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>92.43%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>92.22%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>92.16%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>10.70 ms</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>93.5</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>N/A (5,448 feats)</td>
              </tr>

              {/* Custom CNN */}
              <tr>
                <td>
                  <strong style={{ color: 'var(--text-primary)' }}>Custom 4-Stage CNN</strong>
                </td>
                <td>Deep Learning (Scratch)</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan-primary)', fontWeight: 700 }}>94.07%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>94.11%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>94.07%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan-primary)', fontWeight: 700 }}>94.03%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>16.88 ms</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>59.2</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>422,086</td>
              </tr>

              {/* EfficientNet Frozen */}
              <tr>
                <td>
                  <strong style={{ color: 'var(--text-primary)' }}>EfficientNetB0 (Frozen)</strong>
                </td>
                <td>Transfer Learning</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>98.15%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>98.25%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>98.15%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>98.15%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>104.47 ms</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>9.6</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>4,214,313 (164K train)</td>
              </tr>

              {/* EfficientNet Fine-Tuned */}
              <tr style={{ background: 'rgba(6, 182, 212, 0.06)' }}>
                <td>
                  <strong style={{ color: 'var(--cyan-primary)' }}>EfficientNetB0 (Fine-Tuned)</strong>
                  <span className="badge-champion">Champion</span>
                </td>
                <td>Transfer Learning + Fine-Tuning</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-pass)', fontWeight: 800 }}>98.89%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-pass)' }}>98.96%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-pass)' }}>98.89%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-pass)', fontWeight: 800 }}>98.89%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45.81 ms</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>21.8</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>4,214,313 (1.6M train)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-Class Breakdown & Confusion Matrix Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.4rem' }}>
        {/* Per-Class Table */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <Zap className="w-4 h-4 text-cyan-400" />
              Per-Class Disaggregated Analysis (Fine-Tuned Model)
            </span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Defect Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Test Support</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Crazing</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45</td>
              </tr>
              <tr>
                <td>Inclusion</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45</td>
              </tr>
              <tr>
                <td>Patches</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-pass)', fontWeight: 800 }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45</td>
              </tr>
              <tr>
                <td>Pitted Surface</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--emerald-pass)', fontWeight: 800 }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45</td>
              </tr>
              <tr>
                <td>Rolled-in Scale</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>97.83%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>98.90%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45</td>
              </tr>
              <tr>
                <td>Scratches</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>100.00%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>97.78%</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>98.88%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>45</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Academic Research Findings Summary */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Core PBL Research Findings & RQ Answers
            </span>
          </div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }} className="space-y-3">
            <p>
              <strong>RQ1 (Custom CNN):</strong> The custom 4-stage convolutional neural network successfully reaches <strong>94.07% accuracy</strong> with only 422K parameters, operating at a high throughput of 59.2 FPS.
            </p>
            <p>
              <strong>RQ2 (Transfer Learning):</strong> Initializing with ImageNet weights lifts performance to <strong>98.15%</strong> (an immediate +4.08% improvement), showing that generic spatial low-level edge features readily transfer to industrial strip textures.
            </p>
            <p>
              <strong>RQ3 (Fine-Tuning):</strong> Unfreezing the top 30 convolutional layers while keeping Batch Normalization frozen reaches peak performance of <strong>98.89% Accuracy and Macro F1</strong>.
            </p>
            <p>
              <strong>RQ6 (Efficiency vs. Accuracy):</strong> Custom CNN offers the best edge efficiency (16.8 ms / 59 FPS, 1.5 MB), whereas Fine-Tuned EfficientNet offers the highest defect detection fidelity (98.89%, 45.8 ms).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
