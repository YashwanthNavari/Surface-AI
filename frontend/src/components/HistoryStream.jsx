import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, Filter, Search, ArrowRight } from 'lucide-react';
import StatusBadge from './StatusBadge';
import { API_BASE_URL } from '../config';

export default function HistoryStream({ onSelectInspection }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterClass, setFilterClass] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  const fetchHistory = () => {
    setLoading(true);
    fetch(`${API_BASE_URL}/api/v1/history?limit=100`)
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

  const formatClassName = (name) => {
    if (!name) return '—';
    return name.replace('_', ' ').replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const exportCSV = () => {
    if (history.length === 0) return;
    const headers = ['ID', 'Timestamp', 'Filename', 'Prediction', 'Confidence', 'Consensus_Status', 'Action'];
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
    link.setAttribute('download', `inspection_audit_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredHistory = history.filter((item) => {
    const matchesClass = filterClass === 'ALL' || item.primary_prediction.toLowerCase() === filterClass.toLowerCase();
    const matchesSearch =
      !searchQuery ||
      item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.primary_prediction.toLowerCase().includes(searchQuery.toLowerCase()) ||
      String(item.id).includes(searchQuery);
    return matchesClass && matchesSearch;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Audit history</h1>
          <p className="page-subtitle">
            Immutable chronological event timeline recorded in local SQLite database.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn-secondary" onClick={fetchHistory}>
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
          <button className="btn-secondary" onClick={exportCSV}>
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Filter & Search Toolbar */}
      <div className="surface-card" style={{ padding: '12px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          {/* Category Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Filter className="w-4 h-4 text-slate-400" />
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Filter:</span>
            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-sm)',
                padding: '4px 8px',
                fontSize: '12px',
                color: 'var(--text-primary)',
              }}
            >
              <option value="ALL">All Defect Classes ({history.length})</option>
              <option value="crazing">Crazing</option>
              <option value="inclusion">Inclusion</option>
              <option value="patches">Patches</option>
              <option value="pitted_surface">Pitted Surface</option>
              <option value="rolled-in_scale">Rolled-in Scale</option>
              <option value="scratches">Scratches</option>
            </select>
          </div>

          {/* Search Input */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Search className="w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search filename, defect, ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-sm)',
                padding: '4px 10px',
                fontSize: '12px',
                color: 'var(--text-primary)',
                width: '220px',
              }}
            />
          </div>
        </div>
      </div>

      {/* Event Timeline */}
      {loading ? (
        <div className="surface-card" style={{ textAlign: 'center', padding: '48px 0' }}>
          <div className="spinner" style={{ margin: '0 auto 10px' }} />
          <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Loading event logs...</div>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="surface-card" style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)', fontSize: '13px' }}>
          No inspection events match your query.
        </div>
      ) : (
        <div className="timeline-list">
          {filteredHistory.map((item) => {
            const isAgreed = item.consensus_status === 'CONSENSUS_AGREED';
            return (
              <div key={item.id} className="timeline-row">
                {/* Left: ID + Defect Name + Consensus Badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <span
                    className="mono"
                    style={{
                      fontSize: '12px',
                      fontWeight: 600,
                      color: 'var(--text-muted)',
                      background: 'var(--bg-surface-subtle)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      border: '1px solid var(--border-default)',
                    }}
                  >
                    #{String(item.id).padStart(3, '0')}
                  </span>

                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {formatClassName(item.primary_prediction)}
                      </span>
                      <StatusBadge type={isAgreed ? 'agreed' : 'disagreed'}>
                        {isAgreed ? `${item.consensus_count}/${item.total_models} models agree` : 'Disagreement'}
                      </StatusBadge>
                    </div>

                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      <span className="mono">{item.filename}</span> • {item.timestamp}
                    </div>
                  </div>
                </div>

                {/* Right: Confidence + View Action Button */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ textAlign: 'right' }}>
                    <div className="mono" style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-blue)' }}>
                      {(item.primary_confidence * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>
                      Confidence
                    </div>
                  </div>

                  <button
                    className="btn-secondary"
                    onClick={() => onSelectInspection(item)}
                    style={{ padding: '6px 12px', fontSize: '12px' }}
                  >
                    <span>Re-run representative sample</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
