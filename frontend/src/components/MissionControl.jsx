import React, { useState, useEffect } from 'react';
import { ArrowRight, ShieldCheck, Zap, Activity } from 'lucide-react';
import MetricCard from './MetricCard';
import StatusBadge from './StatusBadge';
import ConfidenceBar from './ConfidenceBar';

export default function MissionControl({ onNavigateWorkspace, recentHistory = [] }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/v1/metrics')
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch((err) => console.error('Error fetching benchmarks:', err));
  }, []);

  const formatClassName = (name) => {
    if (!name) return '—';
    return name.replace('_', ' ').replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  // Model benchmarks data fallback
  const modelsData = metrics?.models || [
    { Model: 'EfficientNetB0 (Fine-Tuned)', Type: 'Transfer Learning', 'Accuracy (%)': 98.89, 'Macro F1 (%)': 98.89, 'Latency (ms)': 25.9, 'Total Parameters': '4.2M', isChampion: true },
    { Model: 'EfficientNetB0 (Frozen)', Type: 'Transfer Learning', 'Accuracy (%)': 98.15, 'Macro F1 (%)': 98.15, 'Latency (ms)': 31.9, 'Total Parameters': '4.2M' },
    { Model: 'Custom 4-Stage CNN', Type: 'From Scratch', 'Accuracy (%)': 94.07, 'Macro F1 (%)': 94.03, 'Latency (ms)': 16.9, 'Total Parameters': '422K', isEdge: true },
    { Model: 'HOG + GLCM + SVM', Type: 'Classical ML', 'Accuracy (%)': 92.22, 'Macro F1 (%)': 92.16, 'Latency (ms)': 10.7, 'Total Parameters': 'N/A' },
  ];

  const defectClasses = [
    { name: 'Crazing', count: 300, share: 16.6 },
    { name: 'Inclusion', count: 300, share: 16.6 },
    { name: 'Patches', count: 300, share: 16.6 },
    { name: 'Pitted Surface', count: 300, share: 16.6 },
    { name: 'Rolled-in Scale', count: 300, share: 16.6 },
    { name: 'Scratches', count: 300, share: 16.6 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Executive overview</h1>
          <p className="page-subtitle">
            Autonomous multi-model surface defect telemetry, classification benchmarks, and inspection distribution.
          </p>
        </div>
        <button className="btn-primary" onClick={() => onNavigateWorkspace('inspect')}>
          <span>Launch inspection</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* 5 Clean KPI Cards */}
      <div className="kpi-grid">
        <MetricCard
          label="Best accuracy"
          value="98.89%"
          subtext="Fine-Tuned EfficientNetB0"
          highlightColor="var(--accent-green)"
        />
        <MetricCard
          label="Custom CNN"
          value="94.07%"
          subtext="Trained from scratch"
          highlightColor="var(--accent-blue)"
        />
        <MetricCard
          label="Champion latency"
          value="25.9 ms"
          subtext="Standard CPU inference"
        />
        <MetricCard
          label="Test samples"
          value="270"
          subtext="Strict evaluation split"
        />
        <MetricCard
          label="Defect classes"
          value="6"
          subtext="NEU benchmark standard"
        />
      </div>

      {/* Main Grid: Model Performance & Accuracy vs Latency Scatter */}
      <div style={{ display: 'grid', gridTemplateColumns: '60% 40%', gap: '24px', alignItems: 'start' }}>
        {/* Model Performance Comparison Table */}
        <div className="surface-card">
          <div className="surface-card-header">
            <div>
              <h2 className="card-title">Model performance comparison</h2>
              <p className="card-subtitle">Empirical evaluation on 270 unseen NEU test images</p>
            </div>
            <StatusBadge type="champion">Authority: 98.89%</StatusBadge>
          </div>

          <table className="clean-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>Latency</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {modelsData.map((m, idx) => {
                const acc = typeof m['Accuracy (%)'] === 'number' ? m['Accuracy (%)'] : parseFloat(m['Accuracy (%)']) || 0;
                const f1 = typeof m['Macro F1 (%)'] === 'number' ? m['Macro F1 (%)'] : parseFloat(m['Macro F1 (%)']) || 0;
                const lat = typeof m['Latency (ms)'] === 'number' ? m['Latency (ms)'].toFixed(1) : m['Latency (ms)'] || '—';
                const isChamp = m.isChampion || acc >= 98.8;
                const isEdge = m.isEdge || m.Model.includes('Custom');

                return (
                  <tr key={idx}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{m.Model}</span>
                        {isChamp && <span className="badge badge-green" style={{ fontSize: '10px', padding: '1px 5px' }}>Best</span>}
                        {isEdge && <span className="badge badge-blue" style={{ fontSize: '10px', padding: '1px 5px' }}>Edge</span>}
                      </div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className="mono" style={{ fontWeight: 600, color: isChamp ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                          {acc.toFixed(1)}%
                        </span>
                        <ConfidenceBar value={acc / 100} isChampion={isChamp} />
                      </div>
                    </td>
                    <td className="mono">{f1.toFixed(1)}%</td>
                    <td className="mono">{lat} ms</td>
                    <td className="mono">{m['Total Parameters']}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Accuracy vs Latency Scatter Chart */}
        <div className="surface-card">
          <div className="surface-card-header">
            <div>
              <h2 className="card-title">Accuracy vs. latency</h2>
              <p className="card-subtitle">Optimal frontier evaluation</p>
            </div>
          </div>

          <div style={{ width: '100%', height: '240px', position: 'relative' }}>
            <svg viewBox="0 0 360 210" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
              {/* Axes and Grid */}
              <line x1="45" y1="20" x2="340" y2="20" stroke="var(--border-default)" strokeDasharray="3" />
              <line x1="45" y1="80" x2="340" y2="80" stroke="var(--border-default)" strokeDasharray="3" />
              <line x1="45" y1="140" x2="340" y2="140" stroke="var(--border-default)" strokeDasharray="3" />
              <line x1="45" y1="180" x2="340" y2="180" stroke="var(--border-strong)" />
              <line x1="45" y1="20" x2="45" y2="180" stroke="var(--border-strong)" />

              {/* Y-Axis Labels (Accuracy %) */}
              <text x="38" y="24" fontSize="10" fill="var(--text-muted)" textAnchor="end" className="mono">100%</text>
              <text x="38" y="84" fontSize="10" fill="var(--text-muted)" textAnchor="end" className="mono">95%</text>
              <text x="38" y="144" fontSize="10" fill="var(--text-muted)" textAnchor="end" className="mono">90%</text>

              {/* X-Axis Labels (Latency ms) */}
              <text x="45" y="196" fontSize="10" fill="var(--text-muted)" textAnchor="middle" className="mono">10ms</text>
              <text x="140" y="196" fontSize="10" fill="var(--text-muted)" textAnchor="middle" className="mono">20ms</text>
              <text x="240" y="196" fontSize="10" fill="var(--text-muted)" textAnchor="middle" className="mono">30ms</text>
              <text x="330" y="196" fontSize="10" fill="var(--text-muted)" textAnchor="middle" className="mono">40ms</text>

              {/* Point 1: HOG + SVM (10.7ms, 92.22%) */}
              <circle cx="52" cy="115" r="5" fill="#6B7280" />
              <text x="58" y="125" fontSize="10" fill="var(--text-secondary)">HOG+SVM (92.2%)</text>

              {/* Point 2: Custom CNN (16.9ms, 94.07%) */}
              <circle cx="110" cy="92" r="5.5" fill="var(--accent-blue)" />
              <text x="116" y="90" fontSize="10" fill="var(--accent-blue)" fontWeight="600">Custom CNN (94.1%)</text>

              {/* Point 3: EfficientNet Frozen (31.9ms, 98.15%) */}
              <circle cx="258" cy="42" r="5" fill="#9CA3AF" />
              <text x="264" y="52" fontSize="10" fill="var(--text-muted)">Frozen (98.2%)</text>

              {/* Point 4: EfficientNet Fine-Tuned (25.9ms, 98.89%) */}
              <circle cx="198" cy="32" r="6.5" fill="var(--accent-green)" />
              <text x="175" y="24" fontSize="10" fill="var(--accent-green)" fontWeight="700">Fine-Tuned (98.9%)</text>
            </svg>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '8px' }}>
            Top-left represents optimal frontier (high accuracy with low latency)
          </div>
        </div>
      </div>

      {/* Lower Row: Defect Distribution & Recent Activity */}
      <div style={{ display: 'grid', gridTemplateColumns: '40% 60%', gap: '24px', alignItems: 'start' }}>
        {/* Defect Class Distribution */}
        <div className="surface-card">
          <div className="surface-card-header">
            <div>
              <h2 className="card-title">Defect class distribution</h2>
              <p className="card-subtitle">Balanced NEU dataset (300 per class)</p>
            </div>
            <span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>1,800 total</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {defectClasses.map((item, idx) => (
              <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                  <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{item.name}</span>
                  <span className="mono" style={{ color: 'var(--text-muted)' }}>{item.count} samples ({item.share}%)</span>
                </div>
                <div style={{ width: '100%', height: '5px', background: 'var(--border-default)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ width: '16.6%', height: '100%', background: 'var(--accent-blue)', borderRadius: '3px' }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Inspections Feed */}
        <div className="surface-card">
          <div className="surface-card-header">
            <div>
              <h2 className="card-title">Recent inspections</h2>
              <p className="card-subtitle">Immutable log records stored in SQLite</p>
            </div>
            <button className="btn-ghost" onClick={() => onNavigateWorkspace('history')} style={{ fontSize: '12px' }}>
              View all history →
            </button>
          </div>

          {recentHistory.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {recentHistory.slice(0, 5).map((item) => {
                const isAgreed = item.consensus_status === 'AGREED' || item.consensus_status === 'CONSENSUS';
                return (
                  <div
                    key={item.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 12px',
                      background: 'var(--bg-surface-subtle)',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-default)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 600 }}>
                        #{String(item.id).padStart(3, '0')}
                      </span>
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {formatClassName(item.primary_prediction)}
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '1px' }}>
                          {item.filename} • {item.timestamp}
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span className="mono" style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-blue)' }}>
                        {(item.primary_confidence * 100).toFixed(1)}%
                      </span>
                      <StatusBadge type={isAgreed ? 'agreed' : 'disagreed'}>
                        {isAgreed ? '3/3 agree' : 'Disagreement'}
                      </StatusBadge>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)', fontSize: '13px' }}>
              No inspections logged yet. Launch an inspection to record results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
