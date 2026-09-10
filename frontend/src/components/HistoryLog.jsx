import React, { useState, useEffect } from 'react';
import { History, CheckCircle2, AlertTriangle, Download, RefreshCw } from 'lucide-react';

export default function HistoryLog() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = () => {
    setLoading(true);
    fetch('http://127.0.0.1:8000/api/v1/history?limit=50')
      .then((res) => res.json())
      .then((data) => {
        setHistory(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching history:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const exportCSV = () => {
    if (history.length === 0) return;
    const headers = ['ID', 'Timestamp', 'Filename', 'Prediction', 'Confidence', 'Consensus_Status', 'Recommendation'];
    const rows = history.map((r) => [
      r.id,
      r.timestamp,
      r.filename,
      r.primary_prediction,
      (r.primary_confidence * 100).toFixed(1) + '%',
      r.consensus_status,
      `"${r.recommendation}"`,
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `inspection_audit_log_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatClassName = (name) => {
    if (!name) return '—';
    return name.replace('_', ' ').replace('-', ' ').toUpperCase();
  };

  return (
    <div className="space-y-4">
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <History className="w-5 h-5 text-cyan-400" />
              SURFACE INSPECTION AUDIT HISTORY
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
              Persistent inspection records stored in local SQLite database (backend/inspections.db).
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.6rem' }}>
            <button className="btn-secondary" onClick={fetchHistory}>
              <RefreshCw className="w-4 h-4" /> Refresh
            </button>
            <button className="btn-secondary" onClick={exportCSV}>
              <Download className="w-4 h-4" /> Export CSV
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '3rem' }}>
            <div className="spinner"></div>
          </div>
        ) : history.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
            No inspection records logged yet. Perform an inspection in the Live Console.
          </div>
        ) : (
          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Audit ID</th>
                  <th>Timestamp</th>
                  <th>Component File</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Consensus Status</th>
                  <th>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row) => (
                  <tr key={row.id}>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      #{String(row.id).padStart(4, '0')}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>{row.timestamp}</td>
                    <td style={{ fontWeight: 600 }}>{row.filename}</td>
                    <td>
                      <strong style={{ color: 'var(--cyan-primary)' }}>
                        {formatClassName(row.primary_prediction)}
                      </strong>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                      {(row.primary_confidence * 100).toFixed(1)}%
                    </td>
                    <td>
                      {row.consensus_status === 'CONSENSUS_AGREED' ? (
                        <span className="history-chip-pass">✓ ALL AGREE</span>
                      ) : (
                        <span className="history-chip-manual">⚠ DISAGREEMENT</span>
                      )}
                    </td>
                    <td style={{ fontSize: '0.82rem' }}>{row.recommendation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
